import { useState, useEffect, useMemo } from 'react'
import PropTypes from 'prop-types'

// Constants
const ANIMATION_DELAY_MS = 100
const ANIMATION_DURATION_MS = 300
const ANIMATION_DURATION_LONG_MS = 500
const ANIMATION_DURATION_GAUGE_MS = 700
const MAX_CORRECTIONS_HEIGHT = 500
const DEFAULT_GAUGE_STROKE_WIDTH = 5
const CONFIDENCE_HIGH_THRESHOLD = 90
const CONFIDENCE_MEDIUM_THRESHOLD = 70
const CONFIDENCE_LOW_THRESHOLD = 50
const MS_PER_SECOND = 1000

// Confidence color configuration
const CONFIDENCE_COLORS = {
  HIGH: { stroke: '#10b981', bg: '#d1fae5' },
  MEDIUM: { stroke: '#3b82f6', bg: '#dbeafe' },
  LOW: { stroke: '#f59e0b', bg: '#fef3c7' },
  VERY_LOW: { stroke: '#ef4444', bg: '#fee2e2' },
}

// Gauge size configuration
const GAUGE_SIZES = {
  sm: { width: 40, stroke: 4, fontSize: 'text-xs' },
  md: { width: 56, stroke: 5, fontSize: 'text-sm' },
  lg: { width: 72, stroke: 6, fontSize: 'text-base' },
}

// Error type labels in Arabic
const ERROR_TYPE_LABELS = {
  hamza: 'همزة',
  hamza_alif: 'همزة الألف',
  hamza_waw: 'همزة الواو',
  hamza_ya: 'همزة الياء',
  taa_marbuta: 'تاء مربوطة',
  alif_maqsura: 'ألف مقصورة',
  alif_maksura: 'ألف مقصورة',
  letter_confusion: 'خلط الحروف',
  spelling: 'إملاء',
  char_swap: 'تبديل الحروف',
  gender_agreement: 'تطابق الجنس',
  number_agreement: 'تطابق العدد',
  morphology: 'صرف',
  agreement: 'تطابق',
  missing_preposition: 'حرف جر مفقود',
  wrong_preposition: 'حرف جر خاطئ',
  preposition: 'حرف جر',
  verb_conjugation: 'تصريف الفعل',
  conjugation: 'تصريف',
  definiteness: 'تنوين/تعريف',
  article: 'أداة التعريف',
  general: 'عام',
}

// Category configuration with icons, colors, and gradients
const CATEGORY_INFO = {
  orthography: {
    label: 'إملاء',
    icon: 'spell',
    color: 'from-red-500 to-rose-600',
    bgGradient: 'from-red-50 to-rose-50',
    borderColor: 'border-red-200',
    textColor: 'text-red-700',
    badgeColor: 'bg-red-100 text-red-800 border-red-200',
    ringColor: 'ring-red-400',
  },
  spelling: {
    label: 'هجاء',
    icon: 'text',
    color: 'from-orange-500 to-amber-600',
    bgGradient: 'from-orange-50 to-amber-50',
    borderColor: 'border-orange-200',
    textColor: 'text-orange-700',
    badgeColor: 'bg-orange-100 text-orange-800 border-orange-200',
    ringColor: 'ring-orange-400',
  },
  morphology: {
    label: 'صرف',
    icon: 'transform',
    color: 'from-blue-500 to-indigo-600',
    bgGradient: 'from-blue-50 to-indigo-50',
    borderColor: 'border-blue-200',
    textColor: 'text-blue-700',
    badgeColor: 'bg-blue-100 text-blue-800 border-blue-200',
    ringColor: 'ring-blue-400',
  },
  syntax: {
    label: 'نحو',
    icon: 'syntax',
    color: 'from-purple-500 to-violet-600',
    bgGradient: 'from-purple-50 to-violet-50',
    borderColor: 'border-purple-200',
    textColor: 'text-purple-700',
    badgeColor: 'bg-purple-100 text-purple-800 border-purple-200',
    ringColor: 'ring-purple-400',
  },
  verb: {
    label: 'فعل',
    icon: 'verb',
    color: 'from-cyan-500 to-teal-600',
    bgGradient: 'from-cyan-50 to-teal-50',
    borderColor: 'border-cyan-200',
    textColor: 'text-cyan-700',
    badgeColor: 'bg-cyan-100 text-cyan-800 border-cyan-200',
    ringColor: 'ring-cyan-400',
  },
  article: {
    label: 'تعريف',
    icon: 'article',
    color: 'from-emerald-500 to-green-600',
    bgGradient: 'from-emerald-50 to-green-50',
    borderColor: 'border-emerald-200',
    textColor: 'text-emerald-700',
    badgeColor: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    ringColor: 'ring-emerald-400',
  },
}

// Map error types to categories
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
  morphology: 'morphology',
  agreement: 'morphology',
  missing_preposition: 'syntax',
  wrong_preposition: 'syntax',
  preposition: 'syntax',
  verb_conjugation: 'verb',
  conjugation: 'verb',
  definiteness: 'article',
  article: 'article',
  general: 'orthography',
}

// SVG Icons for each category
const CategoryIcon = ({ category, className = "w-5 h-5" }) => {
  const icons = {
    spell: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
      </svg>
    ),
    text: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16m-7 6h7" />
      </svg>
    ),
    transform: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
      </svg>
    ),
    syntax: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    verb: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    article: (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
      </svg>
    ),
  }
  return icons[category] || icons.spell
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

function getErrorLabel(errorType) {
  if (ERROR_TYPE_LABELS[errorType]) {
    return ERROR_TYPE_LABELS[errorType]
  }
  for (const [key, label] of Object.entries(ERROR_TYPE_LABELS)) {
    if (errorType.includes(key) || key.includes(errorType)) {
      return label
    }
  }
  return errorType
}

// Animated confidence gauge component
function ConfidenceGauge({ confidence, size }) {
  const [animatedValue, setAnimatedValue] = useState(0)

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(confidence * 100), ANIMATION_DELAY_MS)
    return () => clearTimeout(timer)
  }, [confidence])

  const { width, stroke, fontSize } = GAUGE_SIZES[size]
  const radius = (width - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (animatedValue / 100) * circumference

  const getColor = (value) => {
    if (value >= CONFIDENCE_HIGH_THRESHOLD) return CONFIDENCE_COLORS.HIGH
    if (value >= CONFIDENCE_MEDIUM_THRESHOLD) return CONFIDENCE_COLORS.MEDIUM
    if (value >= CONFIDENCE_LOW_THRESHOLD) return CONFIDENCE_COLORS.LOW
    return CONFIDENCE_COLORS.VERY_LOW
  }

  const colors = getColor(animatedValue)

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={width} height={width} className="transform -rotate-90">
        <circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={stroke}
        />
        <circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className={`transition-all duration-${ANIMATION_DURATION_GAUGE_MS} ease-out`}
        />
      </svg>
      <div className={`absolute ${fontSize} font-bold`} style={{ color: colors.stroke }}>
        {Math.round(animatedValue)}%
      </div>
    </div>
  )
}

ConfidenceGauge.propTypes = {
  confidence: PropTypes.number.isRequired,
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
}

ConfidenceGauge.defaultProps = {
  size: 'md',
}

// Processing time display component
function ProcessingTime({ time }) {
  const displayTime = time < MS_PER_SECOND
    ? `${time.toFixed(0)} مللي`
    : `${(time / MS_PER_SECOND).toFixed(2)} ثانية`

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-slate-50 to-gray-50 rounded-lg border border-slate-200">
      <svg className="w-4 h-4 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span className="text-sm text-slate-600 font-medium">{displayTime}</span>
    </div>
  )
}

ProcessingTime.propTypes = {
  time: PropTypes.number.isRequired,
}

// Model contribution bar chart
function ModelContributionChart({ contributions }) {
  const total = Object.values(contributions).reduce((a, b) => a + b, 0)
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), ANIMATION_DELAY_MS)
    return () => clearTimeout(timer)
  }, [])

  const modelColors = {
    transformer: 'from-blue-500 to-blue-600',
    rule_based: 'from-emerald-500 to-emerald-600',
    morphology: 'from-purple-500 to-purple-600',
    neural: 'from-orange-500 to-orange-600',
    default: 'from-slate-500 to-slate-600',
  }

  const getModelColor = (model) => {
    for (const [key, color] of Object.entries(modelColors)) {
      if (model.toLowerCase().includes(key)) return color
    }
    return modelColors.default
  }

  const formatModelName = (model) => {
    return model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  /**
   * Staggered animation for model contribution bars.
   * Uses inline styles for:
   * - width: Dynamic percentage value based on contribution count
   * - transitionDelay: Index-based delay (idx * 100ms) for cascading reveal effect
   * These values vary per item and cannot be achieved with static CSS classes.
   */
  return (
    <div className="space-y-3">
      {Object.entries(contributions).map(([model, count], idx) => {
        const percentage = (count / total) * 100
        return (
          <div key={model} className="group">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm font-medium text-gray-700 group-hover:text-gray-900 transition-colors">
                {formatModelName(model)}
              </span>
              <span className="text-xs font-semibold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                {count} تصحيح
              </span>
            </div>
            <div className="relative h-2.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className={`absolute inset-y-0 right-0 bg-gradient-to-l ${getModelColor(model)} rounded-full transition-all duration-700 ease-out`}
                style={{
                  width: animated ? `${percentage}%` : '0%',
                  transitionDelay: `${idx * 100}ms`
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

ModelContributionChart.propTypes = {
  contributions: PropTypes.objectOf(PropTypes.number).isRequired,
}

// Collapsible category section
function CategorySection({ category, corrections, categoryInfo, selectedCorrection, onCorrectionClick, isExpanded, onToggle }) {
  const sectionId = `category-${category}-content`

  return (
    <div className="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
      {/* Category Header */}
      <button
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-controls={sectionId}
        aria-label={`${categoryInfo.label}: ${corrections.length} ${corrections.length === 1 ? 'تصحيح' : 'تصحيحات'}`}
        className={`w-full flex items-center justify-between p-4 bg-gradient-to-l ${categoryInfo.bgGradient} hover:brightness-95 transition-all active:scale-[0.99] focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset`}
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-gradient-to-br ${categoryInfo.color} text-white shadow-sm`} aria-hidden="true">
            <CategoryIcon category={categoryInfo.icon} className="w-5 h-5" />
          </div>
          <div className="text-right">
            <h4 className={`font-bold ${categoryInfo.textColor}`}>{categoryInfo.label}</h4>
            <p className="text-xs text-gray-500">{corrections.length} {corrections.length === 1 ? 'تصحيح' : 'تصحيحات'}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`px-3 py-1 rounded-full text-sm font-bold bg-gradient-to-r ${categoryInfo.color} text-white shadow-sm`} aria-hidden="true">
            {corrections.length}
          </div>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Corrections List */}
      <div
        id={sectionId}
        role="region"
        aria-label={`قائمة تصحيحات ${categoryInfo.label}`}
        className={`transition-all duration-300 ease-in-out ${isExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'} overflow-hidden`}
      >
        <div className="p-3 space-y-2 bg-gray-50/50">
          {corrections.map((correction, localIdx) => (
            <ErrorCard
              key={correction.idx}
              correction={correction}
              categoryInfo={categoryInfo}
              isSelected={selectedCorrection === correction.idx}
              onClick={() => onCorrectionClick(correction.idx)}
              animationDelay={localIdx * 50}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

CategorySection.propTypes = {
  category: PropTypes.string.isRequired,
  corrections: PropTypes.arrayOf(
    PropTypes.shape({
      idx: PropTypes.number.isRequired,
      original: PropTypes.string.isRequired,
      corrected: PropTypes.string.isRequired,
      error_type: PropTypes.string,
      confidence: PropTypes.number,
      model: PropTypes.string,
    })
  ).isRequired,
  categoryInfo: PropTypes.shape({
    label: PropTypes.string.isRequired,
    icon: PropTypes.string.isRequired,
    color: PropTypes.string.isRequired,
    bgGradient: PropTypes.string.isRequired,
    borderColor: PropTypes.string.isRequired,
    textColor: PropTypes.string.isRequired,
    badgeColor: PropTypes.string.isRequired,
    ringColor: PropTypes.string.isRequired,
  }).isRequired,
  selectedCorrection: PropTypes.number,
  onCorrectionClick: PropTypes.func.isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
}

CategorySection.defaultProps = {
  selectedCorrection: null,
}

/**
 * Individual error card component with staggered entrance animation.
 *
 * Note: animationDelay prop is used with JavaScript setTimeout (not CSS animation-delay)
 * because the delay varies by index (localIdx * 50ms) to create a cascading reveal effect.
 * Using setTimeout allows for more precise control and doesn't require generating
 * CSS classes for every possible delay value.
 */
function ErrorCard({ correction, categoryInfo, isSelected, onClick, animationDelay }) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), animationDelay)
    return () => clearTimeout(timer)
  }, [animationDelay])

  return (
    <div
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
      role="button"
      tabIndex={0}
      aria-label={`تصحيح: "${correction.original}" إلى "${correction.corrected}"، نوع الخطأ: ${getErrorLabel(correction.error_type)}، الثقة: ${Math.round(correction.confidence * 100)}%`}
      aria-pressed={isSelected}
      className={`
        relative p-4 rounded-xl border-2 cursor-pointer
        transform transition-all duration-300 ease-out
        ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0'}
        ${isSelected
          ? `border-transparent ring-2 ${categoryInfo.ringColor} bg-white shadow-lg scale-[1.02]`
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md hover:scale-[1.01]'
        }
        focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 active:scale-[0.99]
      `}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className={`absolute top-0 right-0 w-0 h-0 border-t-[24px] border-l-[24px] border-t-transparent border-l-transparent rounded-tr-xl`}
          style={{ borderRightWidth: '24px', borderRightColor: categoryInfo.ringColor.includes('red') ? '#f87171' : categoryInfo.ringColor.includes('orange') ? '#fb923c' : categoryInfo.ringColor.includes('blue') ? '#60a5fa' : categoryInfo.ringColor.includes('purple') ? '#a78bfa' : categoryInfo.ringColor.includes('cyan') ? '#22d3d3' : '#34d399' }}
        />
      )}

      <div className="flex items-start justify-between gap-4">
        {/* Error Details */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-3 flex-wrap">
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border ${categoryInfo.badgeColor}`}>
              <CategoryIcon category={categoryInfo.icon} className="w-3.5 h-3.5" />
              {getErrorLabel(correction.error_type)}
            </span>
            <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-md">
              {correction.model?.replace(/_/g, ' ') || 'نموذج'}
            </span>
          </div>

          {/* Original to Corrected */}
          <div className="font-arabic text-lg flex items-center gap-3 flex-wrap" dir="rtl">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-red-50 border border-red-200 rounded-lg">
              <svg className="w-4 h-4 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
              <span className="text-red-600 line-through decoration-2">{correction.original}</span>
            </span>

            <svg className="w-5 h-5 text-gray-400 flex-shrink-0 transform rotate-180" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>

            <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
              <svg className="w-4 h-4 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-green-600 font-medium">{correction.corrected}</span>
            </span>
          </div>
        </div>

        {/* Confidence Gauge */}
        <div className="flex-shrink-0">
          <ConfidenceGauge confidence={correction.confidence} size="md" />
          <div className="text-center mt-1">
            <span className="text-xs text-gray-500">ثقة</span>
          </div>
        </div>
      </div>
    </div>
  )
}

ErrorCard.propTypes = {
  correction: PropTypes.shape({
    idx: PropTypes.number,
    original: PropTypes.string.isRequired,
    corrected: PropTypes.string.isRequired,
    error_type: PropTypes.string,
    confidence: PropTypes.number,
    model: PropTypes.string,
  }).isRequired,
  categoryInfo: PropTypes.shape({
    label: PropTypes.string.isRequired,
    icon: PropTypes.string.isRequired,
    badgeColor: PropTypes.string.isRequired,
    ringColor: PropTypes.string.isRequired,
  }).isRequired,
  isSelected: PropTypes.bool,
  onClick: PropTypes.func.isRequired,
  animationDelay: PropTypes.number,
}

ErrorCard.defaultProps = {
  isSelected: false,
  animationDelay: 0,
}

// Stats card component
function StatsCard({ icon, label, value, subValue, gradient }) {
  return (
    <div className={`relative overflow-hidden p-4 rounded-xl bg-gradient-to-br ${gradient} border border-white/20 shadow-sm`}>
      <div className="absolute top-0 left-0 w-full h-full bg-white/40 backdrop-blur-sm" />
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-1">
          {icon}
          <span className="text-xs font-medium text-gray-600">{label}</span>
        </div>
        <div className="text-2xl font-bold text-gray-800">{value}</div>
        {subValue && <div className="text-xs text-gray-500 mt-0.5">{subValue}</div>}
      </div>
    </div>
  )
}

StatsCard.propTypes = {
  icon: PropTypes.node.isRequired,
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  subValue: PropTypes.string,
  gradient: PropTypes.string.isRequired,
}

StatsCard.defaultProps = {
  subValue: null,
}

function ErrorDetails({ corrections, modelContributions, confidence, processingTime, selectedCorrection, onCorrectionClick }) {
  const [expandedCategories, setExpandedCategories] = useState({})
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    // Auto-expand all categories by default
    const expanded = {}
    Object.keys(CATEGORY_INFO).forEach(cat => { expanded[cat] = true })
    setExpandedCategories(expanded)
  }, [])

  // Group corrections by category
  const correctionsByCategory = useMemo(() => {
    return corrections.reduce((acc, correction, idx) => {
      const category = getErrorCategory(correction.error_type)
      if (!acc[category]) acc[category] = []
      acc[category].push({ ...correction, idx })
      return acc
    }, {})
  }, [corrections])

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }))
  }

  const totalCorrections = corrections.length
  const avgConfidence = corrections.length > 0
    ? corrections.reduce((sum, c) => sum + c.confidence, 0) / corrections.length
    : 0

  if (corrections.length === 0) {
    return (
      <div className={`
        bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg border border-gray-200 p-8
        transition-all duration-500 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}
      `}>
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-green-100 to-emerald-100 flex items-center justify-center">
            <svg className="w-10 h-10 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">النص صحيح تماماً</h3>
          <p className="text-gray-500">لم يتم العثور على أي أخطاء نحوية أو إملائية</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`
      bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg border border-gray-200 overflow-hidden
      transition-all duration-500 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}
    `}>
      {/* Header */}
      <div className="bg-gradient-to-l from-nahawi-primary to-nahawi-secondary p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold mb-1">تفاصيل التصحيحات</h3>
            <p className="text-blue-100 text-sm">تحليل شامل للأخطاء المكتشفة</p>
          </div>
          <div className="flex items-center gap-3">
            {processingTime && <ProcessingTime time={processingTime} />}
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4 p-6 border-b border-gray-200">
        <StatsCard
          icon={<svg className="w-4 h-4 text-red-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>}
          label="إجمالي الأخطاء"
          value={totalCorrections}
          subValue={`في ${Object.keys(correctionsByCategory).length} فئات`}
          gradient="from-red-50 to-rose-50"
        />
        <StatsCard
          icon={<svg className="w-4 h-4 text-blue-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>}
          label="متوسط الثقة"
          value={`${Math.round(avgConfidence * 100)}%`}
          subValue={avgConfidence >= 0.8 ? 'ثقة عالية' : avgConfidence >= 0.6 ? 'ثقة متوسطة' : 'ثقة منخفضة'}
          gradient="from-blue-50 to-indigo-50"
        />
        <StatsCard
          icon={<svg className="w-4 h-4 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>}
          label="النماذج المستخدمة"
          value={Object.keys(modelContributions).length}
          subValue="نماذج تصحيح"
          gradient="from-emerald-50 to-green-50"
        />
      </div>

      {/* Category Legend - Quick Overview */}
      <div className="flex flex-wrap gap-2 p-4 bg-gray-50/50 border-b border-gray-200">
        {Object.entries(CATEGORY_INFO).map(([key, info]) => {
          const count = correctionsByCategory[key]?.length || 0
          if (count === 0) return null
          return (
            <button
              key={key}
              onClick={() => toggleCategory(key)}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium border transition-all
                ${expandedCategories[key] ? info.badgeColor : 'bg-gray-100 text-gray-500 border-gray-200'}
                hover:scale-105 hover:shadow-sm active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1
              `}
            >
              <CategoryIcon category={info.icon} className="w-4 h-4" />
              {info.label}: {count}
            </button>
          )
        })}
      </div>

      {/* Corrections by Category */}
      <div className="p-6 space-y-4 max-h-[500px] overflow-y-auto scrollbar-thin scroll-smooth">
        {Object.entries(CATEGORY_INFO).map(([category, categoryInfo]) => {
          const categoryCorrestions = correctionsByCategory[category]
          if (!categoryCorrestions || categoryCorrestions.length === 0) return null

          return (
            <CategorySection
              key={category}
              category={category}
              corrections={categoryCorrestions}
              categoryInfo={categoryInfo}
              selectedCorrection={selectedCorrection}
              onCorrectionClick={onCorrectionClick}
              isExpanded={expandedCategories[category]}
              onToggle={() => toggleCategory(category)}
            />
          )
        })}
      </div>

      {/* Model Contributions */}
      {Object.keys(modelContributions).length > 0 && (
        <div className="p-6 bg-gradient-to-t from-gray-100 to-gray-50 border-t border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-slate-500 to-slate-600 text-white shadow-sm">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div>
              <h4 className="font-bold text-gray-800">مساهمات النماذج</h4>
              <p className="text-xs text-gray-500">توزيع التصحيحات حسب النموذج</p>
            </div>
          </div>
          <ModelContributionChart contributions={modelContributions} />
        </div>
      )}
    </div>
  )
}

ErrorDetails.propTypes = {
  corrections: PropTypes.arrayOf(
    PropTypes.shape({
      original: PropTypes.string.isRequired,
      corrected: PropTypes.string.isRequired,
      error_type: PropTypes.string,
      confidence: PropTypes.number,
      model: PropTypes.string,
    })
  ).isRequired,
  modelContributions: PropTypes.objectOf(PropTypes.number),
  confidence: PropTypes.number,
  processingTime: PropTypes.number,
  selectedCorrection: PropTypes.number,
  onCorrectionClick: PropTypes.func,
}

ErrorDetails.defaultProps = {
  modelContributions: {},
  confidence: 0,
  processingTime: null,
  selectedCorrection: null,
  onCorrectionClick: () => {},
}

export default ErrorDetails
