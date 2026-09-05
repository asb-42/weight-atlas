# Deployment Runbook — atlas.saga-ai.org (Phase 1 M1)

> Living document (update as the deployment evolves — this is NOT a dated
> proposal). Covers: MariaDB setup, app deploy, Apache TLS + reverse proxy,
> Varnish caching, history migration, seed content, verification, rollback.
> Out of scope here: GitHub OAuth + API keys + quotas (unimplemented code,
> tracked in `docs/2026-09-05_scan-registry-phase1.md` §8/§9 — do NOT expose
> this host publicly without them, see §0).

---

## 0. Read this first — what is and isn't safe today

The app currently has **no authentication**. Until OAuth/API keys land:

- Bind uvicorn to **127.0.0.1 only** and put Apache basic-auth (or an IP
  allowlist) in front of everything except `/healthz`.
- Treat the M1 deployment as a **private beta** even though it is on a
  public domain. The proposal's M2 (public read) requires auth to exist.

`main` at or after the M1 backend commit is required
(`api: MariaDB job-store backend`).

## 1. Host prerequisites

- Debian, root access. Python **3.11+** (`requires-python = ">=3.11"`).
- MariaDB 10.11+ (Debian 12 default is fine — the DDL avoids TEXT
  defaults, which MariaDB does not support).
- Apache2 with `proxy`, `proxy_http`, `headers`, `ssl` modules;
  `certbot` (Let's Encrypt); Varnish (optional but recommended, §5).
- A dedicated unprivileged user, e.g. `atlas`. Never run uvicorn as root.

## 2. MariaDB setup

```sql
-- as root (mariadb CLI):
CREATE DATABASE atlas CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'atlas'@'localhost' IDENTIFIED BY '<STRONG_GENERATED_PASSWORD>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER
  ON atlas.* TO 'atlas'@'localhost';
FLUSH PRIVILEGES;
```

Notes:

- `CREATE`/`ALTER` are needed: the app creates the table and applies
  schema migrations itself on first connect (`init_schema`).
- The app connects over TCP (`127.0.0.1:3306`), not the unix socket —
  keep `skip-networking` OFF (Debian default is on-network localhost).
- Connection string form: `mysql://atlas:<pass>@127.0.0.1:3306/atlas`
- Include the MariaDB data dir in the existing host backup routines
  (plus `mariadb-dump atlas` on a schedule once there is real data).

## 3. App install

```bash
# as atlas user (or root for /opt — then chown to atlas):
git clone <repo-url> /opt/weight-atlas
cd /opt/weight-atlas && git checkout <pinned-commit-sha>   # pin, don't track main
python3 -m venv .venv
.venv/bin/pip install ".[mysql]"
```

Verify the install before wiring anything:

```bash
.venv/bin/python -c "import weight_atlas.api.store, pymysql; print('ok')"
.venv/bin/weight-atlas --help | grep -E "db-copy|serve"
```

Python 3.11 minimum; 3.12 recommended (matches dev/CI).

## 4. Seed content — use packages, NOT db-copy

`weight-atlas db-copy` transfers job rows verbatim, **including absolute
`out_dir` paths**. Dev-machine paths (`/media/data/...`) do not exist on
the server, so do NOT copy dev history. Instead, for each seed scan:

```bash
# on the machine holding the scan:
weight-atlas export /path/to/scan --out scan.wasc --profile full
# copy scan.wasc to the server, then on the server:
weight-atlas import scan.wasc --out /srv/atlas/scans/<name>
# register (or POST /api/import {"scan_dir": "/srv/atlas/scans/<name>"})
```

`db-copy` is for same-host backend switches (SQLite file → MariaDB URL on
one machine), not for seeding a fresh host.

Directory layout on the server (suggested):

```
/srv/atlas/
  scans/        # scan artefact dirs (imported packages land here)
  packages/     # received .wasc files (output_root/packages default)
  venv + repo checkout
```

The app needs write access to the scan dirs, `packages/`, and its log
dir — `chown -R atlas:atlas /srv/atlas`.

## 5. systemd unit

`/etc/systemd/system/weight-atlas.service`:

```ini
[Unit]
Description=weight-atlas web UI (atlas.saga-ai.org)
After=network.target mariadb.service
Requires=mariadb.service

[Service]
User=atlas
Group=atlas
WorkingDirectory=/opt/weight-atlas
EnvironmentFile=/etc/weight-atlas/env
ExecStart=/opt/weight-atlas/.venv/bin/weight-atlas serve \
  --host 127.0.0.1 --port 8000 \
  --proxy-headers --forwarded-allow-ips 127.0.0.1
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

`/etc/weight-atlas/env` (root-owned, `0600`):

```ini
WEIGHT_ATLAS_DB_URL=mysql://atlas:<pass>@127.0.0.1:3306/atlas
WEIGHT_ATLAS_OUTPUT_ROOT=/srv/atlas/scans
```

Notes:

- `--proxy-headers` is **required** behind Apache (correct scheme/host
  for redirects and future OAuth callbacks); `--forwarded-allow-ips`
  must list ONLY the proxy (spoofed `X-Forwarded-*` from anyone else
  would lie about the scheme).
- `WEIGHT_ATLAS_OUTPUT_ROOT` pins where scan/import output lands.
- `systemctl daemon-reload && systemctl enable --now weight-atlas`,
  then `curl -s http://127.0.0.1:8000/healthz` → `{"status":"ok",...}`.

## 6. Apache: TLS + reverse proxy (+ interim access control)

```apache
<VirtualHost *:443>
    ServerName atlas.saga-ai.org

    SSLEngine on
    # ... certbot-managed SSLCertificateFile/KeyFile lines ...

    ProxyPreserveHost On
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-Host "atlas.saga-ai.org"
    ProxyPass / http://127.0.0.1:8000/ connectiontimeout=10 timeout=600
    ProxyPassReverse / http://127.0.0.1:8000/

    # ---- INTERIM, remove once GitHub OAuth + API keys land ----
    # The app has no auth yet. Until then, gate everything except
    # the health check behind basic auth (or an IP allowlist).
    <Location />
        AuthType Basic
        AuthName "atlas-beta"
        AuthUserFile /etc/weight-atlas/htpasswd
        Require valid-user
    </Location>
    <Location /healthz>
        Require all granted
    </Location>
    # ---- end interim ----
</VirtualHost>
```

- `certbot --apache -d atlas.saga-ai.org` for TLS; redirect port 80 → 443.
- `timeout=600`: the import endpoint renders sheets synchronously
  ("potentially minutes"); normal browsing needs far less, but one slow
  import must not 502 the UI.
- Create the htpasswd file with `htpasswd -c` (apache2-utils); delete
  this block only when real auth ships.

## 7. Varnish (recommended, optional for M1)

Sheet PNGs and gallery assets are immutable once written — ideal cache
material. Minimal VCL posture (adapt to the local Varnish version):

- Cache: `GET` `/models/*/artifacts/render/*`, `/static/*` — long TTL.
- **Pass** (never cache): everything under `/api/*`, all POSTs, anything
  with `Authorization` or session cookies (future-proofing for OAuth).
- Backend health probe: `GET /healthz` expecting 200 (not 503 — a
  degraded DB must pull the backend out of rotation).

Skip Varnish only if the Apache layer already covers caching needs;
do not skip the `/healthz` probe wherever the health check lives.

## 8. Verification checklist (run top to bottom after deploy)

1. `curl -s http://127.0.0.1:8000/healthz` → `{"status":"ok","database":"ok"}`
2. `curl -sk https://atlas.saga-ai.org/healthz` → same, over TLS.
3. MariaDB has the table: `SHOW TABLES;` → `jobs`; row count 0 on fresh.
4. Import one seed package (full profile), open its model page, confirm
   sheets + records + scatter render.
5. `weight-atlas db-copy --from <sqlite> --to <url>` — exercise once
   against a scratch MariaDB database to prove the migration path
   (even though prod seeds via packages, §4).
6. Restart MariaDB while the app runs → `/healthz` must flip to 503,
   then recover to 200 without restarting the app (reconnect is
   per-operation).
7. Confirm basic-auth (or allowlist) gates `/`, and `/healthz` stays open.
8. Backup dry-run: `mariadb-dump atlas` restores into a scratch DB.

## 9. Rollback

- The SQLite path is untouched: unsetting `WEIGHT_ATLAS_DB_URL` (or
  pointing it at nothing and restarting) returns the app to
  `WEIGHT_ATLAS_DB_PATH` behavior. Note the asymmetry: jobs created in
  MariaDB do not exist in SQLite — export anything worth keeping as
  `.wasc` before rolling back.
- Keep the previous checkout directory (`/opt/weight-atlas.prev`) until
  the new one passes §8.

## 10. What is deliberately NOT in this runbook

- GitHub OAuth, API keys, quotas, moderation UI — unimplemented code
  (next work item after this runbook; the M1 milestone is not done
  until auth exists — the §6 interim gate is load-bearing until then).
- Postgres — the host runs MariaDB; do not introduce a second database
  system.
- Public upload hardening at scale (archive-bomb limits beyond the
  existing guards, per-key quotas) — Phase 1 M3 scope.
