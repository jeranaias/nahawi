import { useState, useCallback } from 'react'
import Editor from './components/Editor'
import CorrectionPanel from './components/CorrectionPanel'
import ErrorDetails from './components/ErrorDetails'
import Header from './components/Header'

// Demo sentences for quick testing
const DEMO_SENTENCES = [
  "اعلنت الحكومه عن خطه جديده لتطوير البنيه التحتيه في المناطق الريفيه",
  "الطالبات المتفوقين حصلوا على منح دراسيه من الجامعه",
  "تعتبر هده المنطقه من اكثر المناطق ازدهارا في الدوله",
  "يبحث الطلاب المعلومات في المكتبه الرقميه",
  "المدينه الكبير تحتاج الى بنيه تحتيه متطوره"
]

// Icon components for stats cards (decorative, hidden from screen readers)
const CorrectionIcon = () => (
  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
  </svg>
)

const ConfidenceIcon = () => (
  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const ModelIcon = () => (
  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
  </svg>
)

const SpeedIcon = () => (
  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
)

function App() {
  const [inputText, setInputText] = useState('')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedCorrection, setSelectedCorrection] = useState(null)
  const [copied, setCopied] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')

  const handleCorrect = useCallback(async () => {
    if (!inputText.trim()) {
      setError('الرجاء إدخال نص للتصحيح')
      return
    }

    setIsLoading(true)
    setError(null)
    setSelectedCorrection(null)

    setStatusMessage('جاري تحليل النص...')

    try {
      const response = await fetch('/api/correct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          strategy: 'cascading',
          confidence_threshold: 0.7
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setStatusMessage(`تم العثور على ${data.corrections?.length || 0} تصحيح`)
    } catch (err) {
      console.error('Correction error:', err)
      setError('حدث خطأ أثناء التصحيح. تأكد من تشغيل الخادم.')
    } finally {
      setIsLoading(false)
    }
  }, [inputText])

  const handleDemoClick = (sentence) => {
    setInputText(sentence)
    setResult(null)
    setError(null)
  }

  const handleCopy = () => {
    if (result?.corrected) {
      navigator.clipboard.writeText(result.corrected)
      setCopied(true)
      setStatusMessage('تم نسخ النص المصحح')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleClear = () => {
    setInputText('')
    setResult(null)
    setError(null)
    setSelectedCorrection(null)
    setStatusMessage('تم مسح النص')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden" lang="ar">
      {/* Skip to main content link for keyboard users */}
      <a
        href="#main-editor"
        className="skip-link"
      >
        تخطي إلى المحتوى الرئيسي
      </a>

      {/* Screen reader live region for status announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {statusMessage}
      </div>

      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-indigo-500/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -left-20 w-60 h-60 bg-gradient-to-br from-emerald-400/15 to-teal-500/15 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-1/4 w-40 h-40 bg-gradient-to-br from-purple-400/10 to-pink-500/10 rounded-full blur-2xl" />
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px]" />
      </div>

      <Header />

      <main id="main-editor" className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12" tabIndex="-1">
        {/* Demo Sentences Section */}
        <nav className="mb-8" aria-label="جمل تجريبية للاختبار السريع">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-slate-300 to-transparent" aria-hidden="true" />
            <span className="text-sm font-medium text-slate-500 px-3" id="demo-sentences-label">جرب أمثلة سريعة</span>
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-slate-300 to-transparent" aria-hidden="true" />
          </div>
          <div className="flex flex-wrap justify-center gap-3" role="group" aria-labelledby="demo-sentences-label">
            {DEMO_SENTENCES.map((sentence, idx) => (
              <button
                key={idx}
                onClick={() => handleDemoClick(sentence)}
                className="group relative px-4 py-2.5 text-sm bg-white/70 backdrop-blur-sm border border-slate-200/80 rounded-xl hover:bg-white hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100/50 transition-all duration-300 max-w-xs overflow-hidden active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
                aria-label={`تحميل مثال: ${sentence}`}
              >
                <span className="relative z-10 text-slate-600 group-hover:text-slate-800 transition-colors" aria-hidden="true">
                  {sentence.slice(0, 30)}...
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-50 to-indigo-50 opacity-0 group-hover:opacity-100 transition-opacity duration-300" aria-hidden="true" />
              </button>
            ))}
          </div>
        </nav>

        {/* Main Editor Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 mb-8">
          {/* Input Panel - Glass morphism card */}
          <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl shadow-slate-200/50 border border-white/80 p-6 sm:p-8 transition-all duration-500 hover:shadow-2xl hover:shadow-slate-300/50">
            {/* Card accent gradient */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-t-2xl" />

            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold text-slate-800">النص الأصلي</h2>
              </div>
              <div className="px-3 py-1.5 bg-slate-100 rounded-lg">
                <span className="text-sm font-medium text-slate-500">
                  {inputText.length} <span className="text-slate-400">حرف</span>
                </span>
              </div>
            </div>

            <Editor
              value={inputText}
              onChange={setInputText}
              placeholder="أدخل النص العربي هنا للتصحيح..."
              disabled={isLoading}
            />

            <div className="flex gap-3 mt-6">
              {/* Premium Correct Button */}
              <button
                onClick={handleCorrect}
                disabled={isLoading || !inputText.trim()}
                aria-label={isLoading ? 'جاري التصحيح' : 'تصحيح النص'}
                aria-busy={isLoading}
                className="group/btn relative flex-1 px-6 py-4 bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 text-white font-semibold rounded-xl overflow-hidden shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 disabled:from-slate-400 disabled:via-slate-400 disabled:to-slate-400 disabled:shadow-none disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] disabled:hover:scale-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2"
              >
                {/* Button shine effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover/btn:translate-x-full transition-transform duration-700 disabled:hidden" aria-hidden="true" />

                {isLoading ? (
                  <span className="relative flex items-center justify-center gap-3">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="tracking-wide">جاري التصحيح...</span>
                  </span>
                ) : (
                  <span className="relative flex items-center justify-center gap-2">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="tracking-wide">تصحيح النص</span>
                  </span>
                )}
              </button>

              <button
                onClick={handleClear}
                aria-label="مسح النص"
                className="px-5 py-4 text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-xl transition-all duration-200 hover:shadow-md active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                <span className="sr-only">مسح النص</span>
              </button>
            </div>
          </div>

          {/* Output Panel - Glass morphism card */}
          <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl shadow-slate-200/50 border border-white/80 p-6 sm:p-8 transition-all duration-500 hover:shadow-2xl hover:shadow-slate-300/50">
            {/* Card accent gradient */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 rounded-t-2xl" />

            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold text-slate-800">النص المصحح</h2>
              </div>

              {result && (
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1.5 bg-emerald-50 text-emerald-600 font-semibold text-sm rounded-lg border border-emerald-200">
                    {result.corrections.length} تصحيح
                  </span>
                  <button
                    onClick={handleCopy}
                    aria-label={copied ? 'تم نسخ النص' : 'نسخ النص المصحح'}
                    className={`group/copy relative px-4 py-2 text-sm font-medium rounded-lg transition-all duration-300 overflow-hidden active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 ${
                      copied
                        ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30'
                        : 'bg-slate-100 text-slate-600 hover:bg-blue-500 hover:text-white hover:shadow-lg hover:shadow-blue-500/30'
                    }`}
                  >
                    <span className="relative z-10 flex items-center gap-2">
                      {copied ? (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          تم النسخ!
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          نسخ
                        </>
                      )}
                    </span>
                  </button>
                </div>
              )}
            </div>

            <CorrectionPanel
              result={result}
              isLoading={isLoading}
              onCorrectionClick={setSelectedCorrection}
              selectedCorrection={selectedCorrection}
            />
          </div>
        </div>

        {/* Error Message - Animated entrance */}
        {error && (
          <div
            role="alert"
            aria-live="assertive"
            className="mb-8 p-5 bg-gradient-to-r from-red-50 to-rose-50 border border-red-200 rounded-2xl shadow-lg shadow-red-100/50 animate-fadeInSlide"
          >
            <div className="flex items-center gap-4">
              <div className="flex-shrink-0 w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center" aria-hidden="true">
                <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-red-700 font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Error Details Panel */}
        {result && result.corrections.length > 0 && (
          <div className="mb-8 animate-fadeInSlide-400">
            <ErrorDetails
              corrections={result.corrections}
              modelContributions={result.model_contributions}
              confidence={result.confidence}
              processingTime={result.processing_time_ms}
              selectedCorrection={selectedCorrection}
              onCorrectionClick={setSelectedCorrection}
            />
          </div>
        )}

        {/* Stats Cards - Premium design */}
        {result && (
          <section
            aria-label="إحصائيات التصحيح"
            className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6 animate-fadeInSlide-500"
          >
            {/* Corrections Card */}
            <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg shadow-blue-100/50 border border-white/80 p-5 sm:p-6 overflow-hidden transition-all duration-500 hover:shadow-xl hover:shadow-blue-200/50 hover:-translate-y-1">
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-blue-500/10 to-indigo-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center mb-4 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform duration-300">
                  <CorrectionIcon />
                  <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  {result.corrections.length}
                </div>
                <div className="text-sm font-medium text-slate-500 mt-1">تصحيحات</div>
              </div>
            </div>

            {/* Confidence Card */}
            <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg shadow-emerald-100/50 border border-white/80 p-5 sm:p-6 overflow-hidden transition-all duration-500 hover:shadow-xl hover:shadow-emerald-200/50 hover:-translate-y-1">
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center mb-4 shadow-lg shadow-emerald-500/30 group-hover:scale-110 transition-transform duration-300">
                  <ConfidenceIcon />
                </div>
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
                  {Math.round(result.confidence * 100)}%
                </div>
                <div className="text-sm font-medium text-slate-500 mt-1">الثقة</div>
              </div>
            </div>

            {/* Models Card */}
            <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg shadow-purple-100/50 border border-white/80 p-5 sm:p-6 overflow-hidden transition-all duration-500 hover:shadow-xl hover:shadow-purple-200/50 hover:-translate-y-1">
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center mb-4 shadow-lg shadow-purple-500/30 group-hover:scale-110 transition-transform duration-300">
                  <ModelIcon />
                </div>
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {Object.keys(result.model_contributions).length}
                </div>
                <div className="text-sm font-medium text-slate-500 mt-1">نماذج مستخدمة</div>
              </div>
            </div>

            {/* Speed Card */}
            <div className="group relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg shadow-amber-100/50 border border-white/80 p-5 sm:p-6 overflow-hidden transition-all duration-500 hover:shadow-xl hover:shadow-amber-200/50 hover:-translate-y-1">
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-amber-500/10 to-orange-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center mb-4 shadow-lg shadow-amber-500/30 group-hover:scale-110 transition-transform duration-300">
                  <SpeedIcon />
                </div>
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent">
                  {Math.round(result.processing_time_ms)}
                </div>
                <div className="text-sm font-medium text-slate-500 mt-1">مللي ثانية</div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer - Minimal and elegant */}
      <footer className="relative z-10 mt-16 sm:mt-24" role="contentinfo">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="relative py-8 border-t border-slate-200/60">
            {/* Gradient line accent */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-24 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent" />

            <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-center sm:text-right">
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Nahawi</span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-500 text-sm">نظام تصحيح القواعد العربية</span>
              </div>

              <div className="flex items-center gap-3">
                <div className="px-3 py-1.5 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-lg">
                  <span className="text-sm font-semibold text-emerald-700">78.84% F0.5</span>
                </div>
                <span className="text-xs text-slate-400">QALB-2014</span>
              </div>
            </div>
          </div>
        </div>
      </footer>

    </div>
  )
}

export default App
