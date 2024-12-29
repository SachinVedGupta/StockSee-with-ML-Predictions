import { spawn } from 'child_process';
import path from 'path';

export default function handler(req, res) {
  if (req.method === 'POST') {
    const { date } = req.body;

    // Adjust the path to the Python script as necessary
    const scriptPath = path.join(process.cwd(), 'scripts', 'getPrices.py');
    const pythonProcess = spawn('python3', [scriptPath, date]);

    pythonProcess.stdout.on('data', (data) => {
      res.status(200).json(JSON.parse(data.toString()));
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
      res.status(500).json({ error: 'Failed to run Python script' });
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
