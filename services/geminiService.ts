
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
            description: "The highly accurate timestamp when the segment starts, including milliseconds (e.g., '00:00:05.123'). Do not include units like 's'.",
          },
          endTime: {
            type: Type.STRING,
            description: "The highly accurate timestamp when the segment ends, including milliseconds (e.g., '00:00:08.456'). Do not include units like 's'.",
          },
          text: {
            type: Type.STRING,
            description: "The transcribed text for this segment.",
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
              text: "Transcribe the following audio file. Provide extremely detailed timestamps including milliseconds for every segment. Ensure the segments are in a clear chronological order. Output the result in JSON format only.",
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
              text: `Translate the following transcription segments into ${targetLanguage}. 
              Maintain the exact same high-precision timestamps (including milliseconds).
              Return the data in the same JSON structure, but add a new 'translatedText' field to each object.
              
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
