
import React, { useEffect, useRef, useState } from 'react';
import { TranscriptionSegment } from '../types';
import { generateSpeech } from '../services/geminiService';
import { decodeBase64, decodeAudioData } from '../utils/audio';

interface SegmentItemProps {
  segment: TranscriptionSegment;
  isActive?: boolean;
  isManualSeek?: boolean;
  onSelect: (startTime: string) => void;
}

const SegmentItem: React.FC<SegmentItemProps> = ({ segment, isActive, isManualSeek, onSelect }) => {
  const elementRef = useRef<HTMLButtonElement>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (isActive && elementRef.current && !isManualSeek) {
      const element = elementRef.current;
      const container = element.closest('.overflow-y-auto');
      
      if (container) {
        const scrollTimeout = window.setTimeout(() => {
          const containerHeight = container.clientHeight;
          const elementTop = element.offsetTop;
          const elementHeight = element.clientHeight;
          const scrollTop = container.scrollTop;

          const buffer = 60; 
          const isFullyVisible = (elementTop >= scrollTop + buffer) && 
                                (elementTop + elementHeight <= scrollTop + containerHeight - buffer);

          if (!isFullyVisible) {
            const targetScrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2);
            container.scrollTo({
              top: targetScrollTop,
              behavior: 'smooth'
            });
          }
        }, 50);
        return () => window.clearTimeout(scrollTimeout);
      }
    }
  }, [isActive, isManualSeek]);

  const handleSpeak = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isSpeaking) return;
    if (!segment.translatedText) return;

    setIsSpeaking(true);
    try {
      const audioData = await generateSpeech(segment.translatedText);
      if (!audioData) throw new Error("No audio data");

      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const ctx = audioContextRef.current;
      const decodedBytes = decodeBase64(audioData);
      const audioBuffer = await decodeAudioData(decodedBytes, ctx, 24000, 1);
      
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => setIsSpeaking(false);
      source.start();
    } catch (error) {
      console.error("TTS error:", error);
      setIsSpeaking(false);
    }
  };

  return (
    <button
      ref={elementRef}
      type="button"
      onClick={(e) => {
        e.preventDefault();
        onSelect(segment.startTime);
      }}
      className={`w-full text-left flex flex-col px-3 py-1.5 transition-all focus:outline-none relative select-none border-l-4 ${
        isActive 
          ? 'bg-blue-50/90 border-blue-600 shadow-sm z-10' 
          : 'hover:bg-slate-50/80 border-transparent text-slate-500'
      }`}
    >
      <div className="flex items-center gap-2 mb-0.5 pointer-events-none">
        <span className={`text-[10px] font-black font-mono px-1.5 py-0.5 rounded tracking-tighter ${
          isActive ? 'text-white bg-blue-600' : 'text-blue-500 bg-blue-100/30'
        }`}>
          {segment.startTime}
        </span>
        {isActive && (
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>
        )}
      </div>
      
      {/* Transcription Text - Big font, tight spacing */}
      <p className={`text-xl leading-tight transition-all duration-150 ${
        isActive ? 'text-slate-900 font-bold' : 'text-slate-800 font-medium'
      }`}>
        {segment.text}
      </p>

      {segment.translatedText && (
        <div className={`mt-1.5 p-2 rounded-lg border flex items-start gap-3 transition-all ${
          isActive ? 'bg-white/95 border-indigo-200 shadow-sm' : 'bg-slate-50/50 border-slate-100'
        }`}>
          <p className={`text-base italic flex-1 leading-tight ${
            isActive ? 'text-indigo-800 font-bold' : 'text-slate-600'
          }`}>
            {segment.translatedText}
          </p>
          <button 
            type="button"
            onClick={handleSpeak}
            disabled={isSpeaking}
            className={`p-1 rounded-md transition-all shadow-sm ${
              isSpeaking ? 'bg-indigo-600 text-white' : 'bg-indigo-100/50 text-indigo-600 hover:bg-indigo-100'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
          </button>
        </div>
      )}
    </button>
  );
};

export default SegmentItem;
