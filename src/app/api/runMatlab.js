import { exec } from 'child_process';
import path from 'path';

export default function handler(req, res) {
  const scriptPath = path.join(process.cwd(), 'runMatlab.js');

  exec(`node ${scriptPath}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing MATLAB script: ${error}`);
      res.status(500).json({ error: 'Failed to run MATLAB script' });
      return;
    }
    console.log(`MATLAB Output: ${stdout}`);
    if (stderr) {
      console.error(`MATLAB Error Output: ${stderr}`);
    }
    res.status(200).json({ message: 'MATLAB script executed successfully' });
  });
}
