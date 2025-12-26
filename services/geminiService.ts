
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
            description: "Timestamp mulai. Format: HH:MM:SS.mmm. Harus sangat presisi terhadap awal suara.",
          },
          endTime: {
            type: Type.STRING,
            description: "Timestamp akhir. Format: HH:MM:SS.mmm. Harus sangat presisi terhadap akhir suara.",
          },
          text: {
            type: Type.STRING,
            description: "Teks hasil transkripsi.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

export async function transcribeAudio(
  modelName: string,
  audioBase64: string,
  mimeType: string
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    
    // Instruksi agresif untuk presisi milidetik
    const precisionInstruction = isGemini3 
      ? "ANALYZE audio wave patterns deeply. DO NOT round timestamps. We need high-fidelity millisecond precision (mmm). If a word starts at 1.234s, mark it as 00:00:01.234, not 00:00:01.200 or 00:00:01.000." 
      : "Provide accurate timestamps with millisecond precision.";

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
              text: `Transcribe this audio. ${precisionInstruction} 
              Return JSON with 'segments'. Use EXACTLY HH:MM:SS.mmm format for all timestamps. 
              Accuracy of the milliseconds is critical for synchronization.`,
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
    return parsed.segments || [];
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
              text: `Translate these segments into ${targetLanguage}. Maintain the original high-precision HH:MM:SS.mmm timestamps exactly.
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
