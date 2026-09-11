const path = require('node:path');
const { spawn } = require('node:child_process');

// Flask does not automatically load Next.js's .env.local file.
require('dotenv').config({ path: path.join(__dirname, '..', '.env.local') });
require('dotenv').config({ path: path.join(__dirname, '.env') });
const child = spawn(path.join(__dirname, '.venv', 'bin', 'python'), ['getPricesFlask.py'], {
  cwd: __dirname, stdio: 'inherit', env: { ...process.env, PORT: process.env.PORT || '5100' }
});
child.on('error', () => {
  console.error('Cannot start Flask. Install scripts/.venv using CODEX_PROJECT.md.');
  process.exitCode = 1;
});
child.on('exit', code => { process.exitCode = code ?? 1; });
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
