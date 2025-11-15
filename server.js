import express from 'express';
import multer from 'multer';
import cors from 'cors';
import fs from 'fs';
import pdfParse from 'pdf-parse';
import Tesseract from 'tesseract.js';
import OpenAI from 'openai';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

function cleanupFile(p) {
  try { fs.unlinkSync(p); } catch (e) { /* ignore */ }
}

app.post('/api/summarize', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const mimetype = req.file.mimetype || '';
    const filepath = req.file.path;
    let extractedText = '';

    if (mimetype === 'application/pdf') {
      const buffer = fs.readFileSync(filepath);
      const data = await pdfParse(buffer);
      extractedText = data.text || '';
    } else if (mimetype.startsWith('image/') || mimetype === 'application/octet-stream') {
      const { data: { text } } = await Tesseract.recognize(filepath, 'eng');
      extractedText = text || '';
    } else {
      cleanupFile(filepath);
      return res.status(400).json({ error: 'Unsupported file type: ' + mimetype });
    }

    cleanupFile(filepath);

    if (!extractedText || !extractedText.trim()) {
      return res.status(400).json({ error: 'No text found in document' });
    }

    const length = req.body.length || req.query.length || 'medium';
    const lengthMap = { short: 150, medium: 350, long: 700 };
    const maxTokens = lengthMap[length] || 350;

    const system = 'You are a helpful assistant that summarizes documents in a clear, human-like tone. Provide key points and a concise summary.';
    const userPrompt = `Please produce a ${length} summary (and list key points) of the following document text:\n\n${extractedText}`;

    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt }
      ],
      max_tokens: maxTokens
    });

    const summary = response.choices?.[0]?.message?.content || 'Summary could not be generated.';

    return res.json({ summary });
  } catch (err) {
    console.error('Processing error:', err);
    return res.status(500).json({ error: 'Processing failed', detail: String(err.message || err) });
  }
});

app.use(express.static(path.join(__dirname, 'dist')));
app.get('*', (req, res) => {
  const index = path.join(__dirname, 'dist', 'index.html');
  if (fs.existsSync(index)) return res.sendFile(index);
  res.status(404).send('Not found');
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server listening on http://localhost:${port}`));
