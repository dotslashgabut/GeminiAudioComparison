
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
            description: "The timestamp when the segment starts. Format: HH:MM:SS.mmm (e.g., '00:00:05.123'). Use high-fidelity millisecond precision based on the exact start of the sound.",
          },
          endTime: {
            type: Type.STRING,
            description: "The timestamp when the segment ends. Format: HH:MM:SS.mmm (e.g., '00:00:08.456'). Capture the precise moment the last word finishes.",
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
    const isGemini3 = modelName.includes('gemini-3');
    
    // Perintah khusus untuk Gemini 3 agar lebih teliti terhadap milidetik
    const precisionInstruction = isGemini3 
      ? "Perform a high-fidelity analysis of the audio frequencies to determine word boundaries. DO NOT round timestamps to the nearest second or tenth. Provide precision to the exact millisecond (mmm)." 
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
              text: `Transcribe this audio with extreme temporal accuracy. ${precisionInstruction} 
              Return a JSON object with 'segments'. Every segment MUST have 'startTime' and 'endTime' in EXACTLY HH:MM:SS.mmm format. 
              Avoid generic timestamps like '.000'.`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: TRANSCRIPTION_SCHEMA,
        temperature: 0.1, // Rendah agar lebih faktual dan presisi
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
              text: `Translate these segments into ${targetLanguage}. Maintain the original high-precision HH:MM:SS.mmm timestamps exactly as provided.
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
