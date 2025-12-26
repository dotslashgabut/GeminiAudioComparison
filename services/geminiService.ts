
import { GoogleGenAI, Type } from "@google/genai";
import { TranscriptionSegment } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const TRANSCRIPTION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          startTime: {
            type: Type.STRING,
            description: "Timestamp mulai. WAJIB format HH:MM:SS.mmm (contoh: '00:00:01.234'). Jangan bulatkan.",
          },
          endTime: {
            type: Type.STRING,
            description: "Timestamp akhir. WAJIB format HH:MM:SS.mmm (contoh: '00:00:04.567').",
          },
          text: {
            type: Type.STRING,
            description: "Teks transkripsi.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Robustly normalizes timestamp strings to HH:MM:SS.mmm
 * Handles cases where models confuse HH:MM:SS with MM:SS:mmm
 */
function normalizeTimestamp(ts: string): string {
  if (!ts) return "00:00:00.000";
  
  // Clean string from any non-digit/colon/period characters
  const clean = ts.replace(/[^\d:.]/g, '');
  
  // Split by both : and . to identify components
  const components = clean.split(/[:.]/);
  
  let hh = "00", mm = "00", ss = "00", mmm = "000";

  if (components.length >= 4) {
    // Likely HH:MM:SS:mmm or HH:MM:SS.mmm
    [hh, mm, ss, mmm] = components;
  } else if (components.length === 3) {
    // Tricky case: HH:MM:SS or MM:SS:mmm?
    // If the 3rd component has 3 digits, it's almost certainly milliseconds
    if (components[2].length === 3) {
      [mm, ss, mmm] = components;
    } else {
      [hh, mm, ss] = components;
    }
  } else if (components.length === 2) {
    // Assume MM:SS
    [mm, ss] = components;
  } else if (components.length === 1) {
    // Assume seconds only
    ss = components[0];
  }

  // Final formatting and padding
  const fHH = hh.padStart(2, '0').substring(0, 2);
  const fMM = mm.padStart(2, '0').substring(0, 2);
  const fSS = ss.padStart(2, '0').substring(0, 2);
  const fMMM = mmm.padEnd(3, '0').substring(0, 3);

  return `${fHH}:${fMM}:${fSS}.${fMMM}`;
}

export async function transcribeAudio(
  modelName: string,
  audioBase64: string,
  mimeType: string
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    
    // Instruksi temporal yang lebih tajam
    const syncInstruction = isGemini3 
      ? "PERINGATAN: Jangan memberikan timestamp mulai sebelum suara vokal benar-benar terdengar (hindari antisipasi). Pastikan 'startTime' selaras dengan milidetik pertama fonem awal kata tersebut."
      : "PENTING: Gunakan format LENGKAP HH:MM:SS.mmm. Contoh: Jika 23 detik, tulis '00:00:23.000', JANGAN tulis '00:23:000'. Pastikan bagian jam (HH) selalu ada.";

    const precisionInstruction = "ANALISIS gelombang suara secara mendetail. JANGAN MEMBULATKAN waktu. Gunakan presisi milidetik (mmm) secara eksplisit. Format WAJIB: HH:MM:SS.mmm.";

    const response = await ai.models.generateContent({
      model: modelName,
      contents: [
        {
          parts: [
            {
              inlineData: {
                data: audioBase64,
                mimeType: mimeType,
              },
            },
            {
              text: `Transkripsikan audio ini. 
              ${precisionInstruction}
              ${syncInstruction}
              
              Format JSON: {"segments": [{"startTime": "HH:MM:SS.mmm", "endTime": "HH:MM:SS.mmm", "text": "..."}]}
              Pastikan konsistensi format waktu agar UI tidak melompat.`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: TRANSCRIPTION_SCHEMA,
        temperature: 0.1,
      },
    });

    const text = response.text;
    if (!text) throw new Error("Empty response from model");

    const parsed = JSON.parse(text);
    const segments = parsed.segments || [];

    return segments.map((s: any) => ({
      startTime: normalizeTimestamp(String(s.startTime)),
      endTime: normalizeTimestamp(String(s.endTime)),
      text: String(s.text)
    }));
  } catch (error: any) {
    console.error(`Error transcribing with ${modelName}:`, error);
    throw new Error(error.message || "Transcription failed");
  }
}

export async function translateSegments(
  segments: TranscriptionSegment[],
  targetLanguage: string
): Promise<TranscriptionSegment[]> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          parts: [
            {
              text: `Translate these segments into ${targetLanguage}. 
              PENTING: JANGAN MENGUBAH angka timestamp sedikitpun. Pertahankan format HH:MM:SS.mmm secara eksak.
              Data: ${JSON.stringify(segments)}`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            segments: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  startTime: { type: Type.STRING },
                  endTime: { type: Type.STRING },
                  text: { type: Type.STRING },
                  translatedText: { type: Type.STRING },
                },
                required: ["startTime", "endTime", "text", "translatedText"],
              },
            },
          },
        },
      },
    });

    const text = response.text;
    if (!text) throw new Error("Empty response from translation model");
    const parsed = JSON.parse(text);
    return parsed.segments || [];
  } catch (error: any) {
    console.error("Translation error:", error);
    throw error;
  }
}

export async function generateSpeech(text: string): Promise<string | undefined> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Zephyr' },
          },
        },
      },
    });

    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  } catch (error) {
    console.error("TTS error:", error);
    throw error;
  }
}
