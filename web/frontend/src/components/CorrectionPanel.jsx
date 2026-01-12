import { useMemo, useState } from 'react'
import PropTypes from 'prop-types'

// Constants
const MIN_PANEL_HEIGHT = '12rem'
const SHIMMER_DELAYS = [0, 100, 200, 300]
const TOOLTIP_MIN_WIDTH = 180
const ANIMATION_DURATION_MS = 200
const PULSE_ANIMATION_DURATION_MS = 1000
const PULSING_DOTS_COUNT = 5
const PULSING_DOT_DELAY_MS = 150

// Map error types to categories for coloring
const ERROR_CATEGORIES = {
  hamza: 'orthography',
  hamza_alif: 'orthography',
  hamza_waw: 'orthography',
  hamza_ya: 'orthography',
  taa_marbuta: 'orthography',
  alif_maqsura: 'orthography',
  alif_maksura: 'orthography',
  letter_confusion: 'spelling',
  spelling: 'spelling',
  char_swap: 'spelling',
  gender_agreement: 'morphology',
  number_agreement: 'morphology',
  agreement: 'morphology',
  morphology: 'morphology',
  missing_preposition: 'syntax',
  wrong_preposition: 'syntax',
  preposition: 'syntax',
  verb_conjugation: 'verb',
  conjugation: 'verb',
  definiteness: 'article',
  article: 'article',
  general: 'orthography',
}

// Category metadata for styling and labels
const CATEGORY_META = {
  orthography: {
    label: 'إملائي',
    bgClass: 'bg-gradient-to-r from-red-50 to-rose-50',
    borderClass: 'border-red-400',
    hoverBgClass: 'hover:bg-red-100',
    dotClass: 'bg-red-500',
    underlineClass: 'decoration-red-400',
    glowClass: 'shadow-red-200',
    badgeClass: 'bg-red-100 text-red-700 border-red-200',
  },
  spelling: {
    label: 'تهجئة',
    bgClass: 'bg-gradient-to-r from-orange-50 to-amber-50',
    borderClass: 'border-orange-400',
    hoverBgClass: 'hover:bg-orange-100',
    dotClass: 'bg-orange-500',
    underlineClass: 'decoration-orange-400',
    glowClass: 'shadow-orange-200',
    badgeClass: 'bg-orange-100 text-orange-700 border-orange-200',
  },
  morphology: {
    label: 'صرفي',
    bgClass: 'bg-gradient-to-r from-blue-50 to-indigo-50',
    borderClass: 'border-blue-400',
    hoverBgClass: 'hover:bg-blue-100',
    dotClass: 'bg-blue-500',
    underlineClass: 'decoration-blue-400',
    glowClass: 'shadow-blue-200',
    badgeClass: 'bg-blue-100 text-blue-700 border-blue-200',
  },
  syntax: {
    label: 'نحوي',
    bgClass: 'bg-gradient-to-r from-purple-50 to-violet-50',
    borderClass: 'border-purple-400',
    hoverBgClass: 'hover:bg-purple-100',
    dotClass: 'bg-purple-500',
    underlineClass: 'decoration-purple-400',
    glowClass: 'shadow-purple-200',
    badgeClass: 'bg-purple-100 text-purple-700 border-purple-200',
  },
  verb: {
    label: 'فعلي',
    bgClass: 'bg-gradient-to-r from-cyan-50 to-teal-50',
    borderClass: 'border-cyan-400',
    hoverBgClass: 'hover:bg-cyan-100',
    dotClass: 'bg-cyan-500',
    underlineClass: 'decoration-cyan-400',
    glowClass: 'shadow-cyan-200',
    badgeClass: 'bg-cyan-100 text-cyan-700 border-cyan-200',
  },
  article: {
    label: 'تعريف',
    bgClass: 'bg-gradient-to-r from-emerald-50 to-green-50',
    borderClass: 'border-emerald-400',
    hoverBgClass: 'hover:bg-emerald-100',
    dotClass: 'bg-emerald-500',
    underlineClass: 'decoration-emerald-400',
    glowClass: 'shadow-emerald-200',
    badgeClass: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  },
}

function getErrorCategory(errorType) {
  if (ERROR_CATEGORIES[errorType]) {
    return ERROR_CATEGORIES[errorType]
  }
  for (const [key, category] of Object.entries(ERROR_CATEGORIES)) {
    if (errorType.includes(key) || key.includes(errorType)) {
      return category
    }
  }
  return 'orthography'
}

/**
 * Shimmer skeleton component for loading states.
 * Maps standard delay values (0, 100, 200, 300ms) to CSS utility classes.
 */
function ShimmerLine({ width, delay }) {
  // Map delay values to CSS animation-delay classes (defined in index.css)
  const delayClassMap = {
    0: '',
    100: 'animation-delay-100',
    200: 'animation-delay-200',
    300: 'animation-delay-300',
  }

  const delayClass = delayClassMap[delay] ?? ''

  return (
    <div
      className={`${width} h-5 rounded-md bg-gradient-to-l from-gray-100 via-gray-200 to-gray-100 bg-[length:200%_100%] animate-shimmer ${delayClass}`}
    />
  )
}

ShimmerLine.propTypes = {
  width: PropTypes.string,
  delay: PropTypes.number,
}

ShimmerLine.defaultProps = {
  width: 'w-full',
  delay: 0,
}

// Loading state component
function LoadingState() {
  return (
    <div
      className="min-h-[12rem] p-6"
      dir="rtl"
      role="status"
      aria-busy="true"
      aria-label="جاري تحليل النص وتصحيحه"
    >
      {/* Shimmer skeleton lines */}
      <div className="space-y-4" aria-hidden="true">
        <ShimmerLine width="w-full" delay={0} />
        <ShimmerLine width="w-11/12" delay={100} />
        <ShimmerLine width="w-4/5" delay={200} />
        <ShimmerLine width="w-9/12" delay={300} />
      </div>

      {/* Animated processing indicator */}
      <div className="mt-8 flex items-center justify-center gap-3">
        <div className="relative flex items-center gap-2" aria-hidden="true">
          {/* Pulsing circles - uses CSS classes for delays (0, 150, 300, 450, 600ms) */}
          <div className="flex gap-1.5">
            {Array.from({ length: PULSING_DOTS_COUNT }, (_, i) => {
              // Map index to animation-delay CSS class (i * 150ms pattern)
              const delayClasses = ['animation-delay-0', 'animation-delay-150', 'animation-delay-300', 'animation-delay-450', 'animation-delay-600']
              return (
                <div
                  key={i}
                  className={`w-2 h-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 animate-pulse animation-duration-1000 ${delayClasses[i]}`}
                />
              )
            })}
          </div>
        </div>
        <span className="text-sm font-medium text-gray-500 animate-pulse">
          جاري تحليل النص وتصحيحه...
        </span>
      </div>

      {/* Decorative processing animation */}
      <div className="mt-4 flex justify-center" aria-hidden="true">
        <div className="relative w-48 h-1 bg-gray-100 rounded-full overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-400 to-transparent animate-slide" />
        </div>
      </div>
    </div>
  )
}

// Empty state component
function EmptyState() {
  return (
    <div className="min-h-[12rem] flex items-center justify-center p-6">
      <div className="text-center max-w-sm">
        {/* Decorative illustration */}
        <div className="relative mx-auto w-24 h-24 mb-6">
          {/* Background glow */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full blur-xl opacity-60" />

          {/* Main icon container */}
          <div className="relative w-full h-full bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg border border-gray-100 flex items-center justify-center transform transition-transform hover:scale-105 hover:rotate-1">
            {/* Document icon */}
            <svg className="w-12 h-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>

            {/* Decorative pen icon */}
            <div className="absolute -bottom-1 -left-1 w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md flex items-center justify-center transform -rotate-12">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Text content */}
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          جاهز للتصحيح
        </h3>
        <p className="text-gray-400 text-sm leading-relaxed">
          أدخل نصاً عربياً في الحقل أعلاه ثم اضغط على
          <span className="mx-1 px-2 py-0.5 bg-blue-50 text-blue-600 rounded font-medium">تصحيح النص</span>
          لعرض التصحيحات
        </p>

        {/* Decorative dots */}
        <div className="mt-6 flex justify-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-gray-200" />
          <div className="w-1.5 h-1.5 rounded-full bg-gray-300" />
          <div className="w-1.5 h-1.5 rounded-full bg-gray-200" />
        </div>
      </div>
    </div>
  )
}

// Success state (no errors found)
function SuccessState({ text }) {
  return (
    <div className={`min-h-[${MIN_PANEL_HEIGHT}] p-6`}>
      {/* Corrected text display */}
      <div
        className="text-lg font-arabic leading-loose text-gray-800 mb-6 p-4 bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-100"
        dir="rtl"
      >
        {text}
      </div>

      {/* Success banner */}
      <div className="relative overflow-hidden bg-gradient-to-r from-emerald-50 via-green-50 to-teal-50 border border-emerald-200 rounded-xl p-4">
        {/* Decorative background pattern */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-0 left-0 w-32 h-32 bg-gradient-to-br from-green-200 to-transparent rounded-full -translate-x-16 -translate-y-16" />
          <div className="absolute bottom-0 right-0 w-24 h-24 bg-gradient-to-tl from-emerald-200 to-transparent rounded-full translate-x-12 translate-y-12" />
        </div>

        <div className="relative flex items-center gap-4" dir="rtl">
          {/* Success icon with animation */}
          <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-emerald-400 to-green-500 rounded-full flex items-center justify-center shadow-lg shadow-emerald-200">
            <svg className="w-6 h-6 text-white animate-success-check" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          </div>

          <div>
            <h4 className="font-semibold text-emerald-800 text-base">
              ممتاز! النص صحيح
            </h4>
            <p className="text-emerald-600 text-sm mt-0.5">
              لم يتم العثور على أي أخطاء نحوية أو إملائية
            </p>
          </div>

          {/* Decorative sparkles */}
          <div className="absolute left-4 top-2">
            <svg className="w-4 h-4 text-yellow-400 animate-sparkle" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z" />
            </svg>
          </div>
          <div className="absolute left-12 bottom-2">
            <svg className="w-3 h-3 text-yellow-300 animate-sparkle-delayed" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  )
}

SuccessState.propTypes = {
  text: PropTypes.string.isRequired,
}

// Error highlight component with tooltip
function ErrorHighlight({ correction, category, isSelected, onClick, children }) {
  const [isHovered, setIsHovered] = useState(false)
  const meta = CATEGORY_META[category]

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onClick()
    }
  }

  return (
    <span
      role="button"
      tabIndex={0}
      aria-label={`تصحيح: ${correction.original} إلى ${correction.corrected}، نوع الخطأ: ${meta.label}`}
      aria-pressed={isSelected}
      className={`
        relative inline-block cursor-pointer
        px-1 py-0.5 mx-0.5 rounded-md
        border-b-2 ${meta.borderClass}
        ${meta.bgClass}
        ${meta.hoverBgClass}
        transition-all duration-200 ease-out
        ${isSelected ? 'ring-2 ring-blue-400 ring-offset-2 shadow-lg ' + meta.glowClass : ''}
        hover:shadow-md hover:-translate-y-0.5
        focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2
        active:scale-95
        group
      `}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onFocus={() => setIsHovered(true)}
      onBlur={() => setIsHovered(false)}
    >
      {/* Animated underline */}
      <span className={`absolute bottom-0 left-0 right-0 h-0.5 ${meta.dotClass} transform origin-right scale-x-0 group-hover:scale-x-100 transition-transform duration-300`} />

      {/* Text content */}
      <span className="relative">{children}</span>

      {/* Enhanced tooltip */}
      <div
        className={`
          absolute z-50 bottom-full right-1/2 transform translate-x-1/2 mb-3
          transition-all duration-200 ease-out
          ${isHovered ? 'opacity-100 visible translate-y-0' : 'opacity-0 invisible translate-y-2'}
        `}
      >
        <div className="bg-gray-900 text-white rounded-xl shadow-2xl overflow-hidden min-w-[180px]">
          {/* Tooltip header */}
          <div className="px-4 py-2 bg-gradient-to-r from-gray-800 to-gray-900 border-b border-gray-700 flex items-center justify-between">
            <span className={`text-xs px-2 py-0.5 rounded-full border ${meta.badgeClass}`}>
              {meta.label}
            </span>
            <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>

          {/* Tooltip content */}
          <div className="p-4 space-y-3">
            {/* Original (wrong) */}
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <span className="text-red-300 line-through text-sm font-arabic">{correction.original}</span>
            </div>

            {/* Arrow */}
            <div className="flex justify-center">
              <svg className="w-4 h-4 text-gray-500 transform rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </div>

            {/* Corrected */}
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="text-green-300 font-medium text-sm font-arabic">{correction.corrected}</span>
            </div>
          </div>
        </div>

        {/* Tooltip arrow */}
        <div className="absolute top-full right-1/2 transform translate-x-1/2 -mt-1">
          <div className="w-3 h-3 bg-gray-900 transform rotate-45" />
        </div>
      </div>
    </span>
  )
}

ErrorHighlight.propTypes = {
  correction: PropTypes.shape({
    original: PropTypes.string.isRequired,
    corrected: PropTypes.string.isRequired,
    start: PropTypes.number,
    end: PropTypes.number,
    error_type: PropTypes.string,
  }).isRequired,
  category: PropTypes.oneOf(['orthography', 'spelling', 'morphology', 'syntax', 'verb', 'article']).isRequired,
  isSelected: PropTypes.bool,
  onClick: PropTypes.func,
  children: PropTypes.node,
}

ErrorHighlight.defaultProps = {
  isSelected: false,
  onClick: () => {},
  children: null,
}

// Main CorrectionPanel component
function CorrectionPanel({ result, isLoading, onCorrectionClick, selectedCorrection }) {
  // Build highlighted text
  const highlightedContent = useMemo(() => {
    if (!result) return null

    const { corrected, corrections } = result

    if (corrections.length === 0) {
      return <span>{corrected}</span>
    }

    const sortedCorrections = [...corrections].sort((a, b) => a.start - b.start)

    const elements = []
    let lastEnd = 0

    sortedCorrections.forEach((correction, idx) => {
      if (correction.start > lastEnd) {
        elements.push(
          <span key={`text-${idx}`} className="text-gray-800">
            {corrected.slice(lastEnd, correction.start)}
          </span>
        )
      }

      const category = getErrorCategory(correction.error_type)
      const isSelected = selectedCorrection === idx

      elements.push(
        <ErrorHighlight
          key={`correction-${idx}`}
          correction={correction}
          category={category}
          isSelected={isSelected}
          onClick={() => onCorrectionClick(idx)}
        >
          {correction.corrected}
        </ErrorHighlight>
      )

      lastEnd = correction.end
    })

    if (lastEnd < corrected.length) {
      elements.push(
        <span key="text-end" className="text-gray-800">
          {corrected.slice(lastEnd)}
        </span>
      )
    }

    return elements
  }, [result, selectedCorrection, onCorrectionClick])

  // Corrections summary for the legend
  const correctionsSummary = useMemo(() => {
    if (!result || result.corrections.length === 0) return null

    const summary = {}
    result.corrections.forEach(c => {
      const category = getErrorCategory(c.error_type)
      summary[category] = (summary[category] || 0) + 1
    })
    return summary
  }, [result])

  // Loading state
  if (isLoading) {
    return <LoadingState />
  }

  // Empty state
  if (!result) {
    return <EmptyState />
  }

  // Success state (no errors)
  if (result.corrections.length === 0) {
    return <SuccessState text={result.corrected} />
  }

  // Main content with corrections
  return (
    <div className="min-h-[12rem] p-6" aria-live="polite" aria-atomic="false">
      {/* Corrected text with highlights */}
      <div
        className="text-lg font-arabic leading-loose text-gray-800 p-4 bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-100 shadow-sm"
        dir="rtl"
        role="region"
        aria-label="النص المصحح مع التصحيحات المميزة"
      >
        {highlightedContent}
      </div>

      {/* Corrections legend and summary */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4" dir="rtl">
        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3">
          {correctionsSummary && Object.entries(correctionsSummary).map(([category, count]) => (
            <div
              key={category}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${CATEGORY_META[category].badgeClass} border text-xs font-medium`}
            >
              <div className={`w-2 h-2 rounded-full ${CATEGORY_META[category].dotClass}`} />
              <span>{CATEGORY_META[category].label}</span>
              <span className="opacity-60">({count})</span>
            </div>
          ))}
        </div>

        {/* Hint text */}
        <p className="text-xs text-gray-400 flex items-center gap-1.5">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
          انقر على أي تصحيح لعرض التفاصيل
        </p>
      </div>

      {/* Total corrections counter */}
      <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-center gap-2 text-sm text-gray-500">
        <span className="font-semibold text-blue-600">{result.corrections.length}</span>
        <span>تصحيح{result.corrections.length > 1 ? 'ات' : ''} تم اكتشافها</span>
      </div>
    </div>
  )
}

CorrectionPanel.propTypes = {
  result: PropTypes.shape({
    corrected: PropTypes.string.isRequired,
    corrections: PropTypes.arrayOf(
      PropTypes.shape({
        original: PropTypes.string.isRequired,
        corrected: PropTypes.string.isRequired,
        start: PropTypes.number.isRequired,
        end: PropTypes.number.isRequired,
        error_type: PropTypes.string,
      })
    ).isRequired,
  }),
  isLoading: PropTypes.bool,
  onCorrectionClick: PropTypes.func,
  selectedCorrection: PropTypes.number,
}

CorrectionPanel.defaultProps = {
  result: null,
  isLoading: false,
  onCorrectionClick: () => {},
  selectedCorrection: null,
}

export default CorrectionPanel
