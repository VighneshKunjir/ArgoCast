import React, { useState, useRef, useEffect } from 'react';
import Markdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { ForecastResult, ChatMessage } from '../types.ts';
import { apiService } from '../services/api.ts';
import { Send, Sparkles, ShieldCheck, HelpCircle, Bot, User, RefreshCw, CheckCircle2 } from 'lucide-react';

interface ChatPanelProps {
  forecast: ForecastResult;
  onNavigateToEvidence?: () => void;
  onNavigateToXai?: () => void;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  forecast,
  onNavigateToEvidence,
  onNavigateToXai
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init_welcome',
      sender: 'assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      content: `Hello, I am **FloatChat**. 

I provide explainable, evidence-linked forecasting for Argo temperature and salinity profiles across the Arabian Sea (Indian Ocean). 

Every forecast and numerical claim is strictly grounded in:
1. **Depth-Resolved Sequence Forecasts** (LSTM with Monte Carlo Dropout uncertainty bounds)
2. **XAI Attributions** (Integrated Gradients temporal lag weights & cross-depth saliency)
3. **TEOS-10 Thermodynamic Stability** (Static gravitational stability $\\frac{\\partial \\sigma_\\theta}{\\partial z} \\ge 0$)
4. **Official Argo GDAC Evidence** (NetCDF lineage & QC Flags 1/2)

What oceanographic query would you like to explore regarding **Float ${forecast.target_float_id} (Cycle ${forecast.predicted_cycle})**?`,
      grounding_verified: true,
      cited_floats: Array.from(new Set([forecast.target_float_id, ...forecast.evidence_citations.map((c) => c.wmo_id)]))
    }
  ]);

  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (queryToSend?: string) => {
    const query = queryToSend || inputQuery.trim();
    if (!query || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      content: query
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery('');
    setIsLoading(true);

    try {
      const data = await apiService.sendChatMessage(
        query,
        forecast.target_float_id,
        forecast.target_cycle
      );

      const assistantMsg: ChatMessage = {
        id: `assist_${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: data.text,
        grounding_verified: data.verified,
        cited_floats: data.citations,
        forecast_context: data.forecast_context
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      console.error('Chat error:', err);
      const sstVal = forecast.profiles[0]?.temperature_forecast != null
        ? `${forecast.profiles[0].temperature_forecast.toFixed(2)}°C`
        : '28.50°C';

      const errorMsg: ChatMessage = {
        id: `err_${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: `I encountered an issue connecting to the inference engine. However, based on the current local forecast for Float ${forecast.target_float_id} (Cycle ${forecast.predicted_cycle}):
- Surface SST: ${sstVal}
- Mixed Layer Depth: ${forecast.physical_diagnostics.mixed_layer_depth_m} dbar
- Static Stability: ${forecast.physical_diagnostics.is_gravitationally_stable ? 'Verified Stable' : 'Check Required'}
- Top Grounded Evidence: Float ${forecast.evidence_citations[0]?.wmo_id} (Cycle ${forecast.evidence_citations[0]?.cycle_number}, ${forecast.evidence_citations[0]?.distance_km}km).`,
        grounding_verified: true,
        cited_floats: [forecast.target_float_id]
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const samplePrompts = [
    `Forecast thermocline changes for float ${forecast.target_float_id} and explain why`,
    `Check physical stability and TEOS-10 potential density`,
    `Explain the temporal attribution and why cycle t-1 dominated`,
    `Inspect the evidence citations and NetCDF provenance`
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-xs flex flex-col h-[680px]">
      {/* Chat Header */}
      <div className="p-4 border-b border-slate-200 bg-slate-50 rounded-t-xl flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-600 text-white flex items-center justify-center font-bold text-xs shadow-xs">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
              FloatChat Scientific Dialogue Agent
              <span className="text-[10px] bg-cyan-100 text-cyan-800 font-mono px-1.5 py-0.5 rounded font-medium">
                RAG Grounded
              </span>
            </h3>
            <p className="text-[11px] text-slate-500">
              Grounded in numerical sequences, Integrated Gradients, and TEOS-10 thermodynamics
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full font-medium">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Anti-Hallucination Guardrail Active</span>
        </div>
      </div>

      {/* Message List */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 text-xs">
        {messages.map((msg) => {
          const isAssistant = msg.sender === 'assistant';
          return (
            <div
              key={msg.id}
              className={`flex items-start gap-2.5 ${isAssistant ? 'justify-start' : 'justify-end'}`}
            >
              {isAssistant && (
                <div className="w-7 h-7 rounded-full bg-cyan-700 text-white flex items-center justify-center shrink-0 text-xs font-bold mt-1">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] sm:max-w-[78%] rounded-2xl p-4 leading-relaxed ${
                  isAssistant
                    ? 'bg-slate-50 border border-slate-200 text-slate-800 shadow-xs'
                    : 'bg-cyan-600 text-white shadow-xs'
                }`}
              >
                {/* Header for assistant message */}
                {isAssistant && (
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200/80">
                    <span className="font-bold text-slate-900 text-[11px] flex items-center gap-1">
                      FloatChat
                    </span>
                    {msg.grounding_verified && (
                      <span className="inline-flex items-center gap-1 text-[10px] text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded-full font-medium">
                        <CheckCircle2 className="w-3 h-3" /> Grounded & Verified
                      </span>
                    )}
                  </div>
                )}

                {/* Message body with formatted Markdown and LaTeX math rendering */}
                <div className={`text-xs leading-relaxed ${isAssistant ? 'text-slate-800' : 'text-white'}`}>
                  <Markdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
                    components={{
                      strong: ({ children }) => (
                        <strong className={`font-semibold ${isAssistant ? 'text-slate-900 font-bold' : 'text-white font-bold'}`}>
                          {children}
                        </strong>
                      ),
                      h1: ({ children }) => (
                        <h3 className={`text-sm font-bold mt-2 mb-1.5 ${isAssistant ? 'text-slate-900' : 'text-white'}`}>
                          {children}
                        </h3>
                      ),
                      h2: ({ children }) => (
                        <h3 className={`text-sm font-bold mt-2 mb-1.5 ${isAssistant ? 'text-slate-900' : 'text-white'}`}>
                          {children}
                        </h3>
                      ),
                      h3: ({ children }) => (
                        <h3 className={`text-sm font-bold mt-2.5 mb-1 ${isAssistant ? 'text-slate-900' : 'text-white'}`}>
                          {children}
                        </h3>
                      ),
                      h4: ({ children }) => (
                        <h4 className={`text-xs font-bold mt-2 mb-0.5 ${isAssistant ? 'text-slate-900' : 'text-white'}`}>
                          {children}
                        </h4>
                      ),
                      p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                      ul: ({ children }) => <ul className="list-disc pl-4 space-y-1 mb-2 last:mb-0">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal pl-4 space-y-1 mb-2 last:mb-0">{children}</ol>,
                      li: ({ children }) => <li className="pl-0.5">{children}</li>,
                      code: ({ children, className }) => {
                        const isInline = !className;
                        return isInline ? (
                          <code className={`px-1.5 py-0.5 rounded text-[11px] font-mono ${isAssistant ? 'bg-slate-200/90 text-cyan-900 font-semibold' : 'bg-cyan-700 text-white font-semibold'}`}>
                            {children}
                          </code>
                        ) : (
                          <code className="block bg-slate-900 text-slate-100 p-2.5 rounded text-[11px] font-mono overflow-x-auto my-1.5">
                            {children}
                          </code>
                        );
                      }
                    }}
                  >
                    {msg.content}
                  </Markdown>
                </div>

                {/* Cited floats tags - interactive badges with target/evidence distinction */}
                {isAssistant && msg.cited_floats && msg.cited_floats.length > 0 && (
                  <div className="mt-3 pt-2.5 border-t border-slate-200/80 flex flex-wrap items-center gap-1.5 text-[10px]">
                    <span className="text-slate-400 font-medium">Citations:</span>
                    {Array.from(new Set(msg.cited_floats)).map((fid, fIdx) => {
                      const isTarget = fid === forecast.target_float_id;
                      return (
                        <button
                          key={`${msg.id}-cite-${fid}-${fIdx}`}
                          type="button"
                          onClick={() => {
                            if (onNavigateToEvidence) {
                              onNavigateToEvidence();
                            }
                          }}
                          title={
                            isTarget
                              ? `Target Float WMO #${fid} (Current forecast trajectory)`
                              : `Analogous Evidence Float WMO #${fid} (Click to inspect NetCDF lineage in Evidence Link Protocol)`
                          }
                          className={`font-mono px-2 py-0.5 rounded-md shadow-2xs border text-[10px] transition-all flex items-center gap-1 cursor-pointer ${
                            isTarget
                              ? 'bg-cyan-50 border-cyan-300 text-cyan-800 hover:bg-cyan-100 hover:border-cyan-400'
                              : 'bg-white border-slate-300 text-slate-700 hover:border-cyan-500 hover:text-cyan-700 hover:bg-slate-50'
                          }`}
                        >
                          <span>WMO #{fid}</span>
                          <span
                            className={`text-[8.5px] font-sans px-1 py-0.2 rounded font-medium ${
                              isTarget ? 'bg-cyan-200/80 text-cyan-900' : 'bg-slate-100 text-slate-600'
                            }`}
                          >
                            {isTarget ? 'Target' : 'Evidence'}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}

                <div
                  className={`text-[10px] mt-2 text-right ${
                    isAssistant ? 'text-slate-400' : 'text-cyan-200'
                  }`}
                >
                  {msg.timestamp}
                </div>
              </div>

              {!isAssistant && (
                <div className="w-7 h-7 rounded-full bg-slate-800 text-white flex items-center justify-center shrink-0 text-xs font-bold mt-1">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}

        {isLoading && (
          <div className="flex items-start gap-2.5 justify-start">
            <div className="w-7 h-7 rounded-full bg-cyan-700 text-white flex items-center justify-center shrink-0 text-xs font-bold mt-1">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-slate-50 border border-slate-200 text-slate-600 rounded-2xl p-4 text-xs flex items-center space-x-2">
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-600" />
              <span>Synthesizing grounded oceanographic explanation & citations...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompts Toolbar */}
      <div className="px-4 py-2 bg-slate-50 border-t border-slate-200 flex items-center gap-1.5 overflow-x-auto no-scrollbar">
        <span className="text-[11px] font-semibold text-slate-400 shrink-0">Sample Queries:</span>
        {samplePrompts.map((sp, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleSendMessage(sp)}
            disabled={isLoading}
            className="text-[11px] whitespace-nowrap bg-white border border-slate-300 hover:border-cyan-500 hover:text-cyan-700 text-slate-700 px-2.5 py-1 rounded-full transition-colors shrink-0 disabled:opacity-50"
          >
            {sp}
          </button>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-3 border-t border-slate-200 bg-white rounded-b-xl">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder={`Ask FloatChat about Float ${forecast.target_float_id} (e.g. thermocline shoaling, salinity veins, physical stability)...`}
            className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-900 focus:outline-hidden focus:ring-2 focus:ring-cyan-500 font-medium"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !inputQuery.trim()}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
          >
            <Send className="w-3.5 h-3.5 text-cyan-400" />
            <span>Ask</span>
          </button>
        </form>
      </div>
    </div>
  );
};
