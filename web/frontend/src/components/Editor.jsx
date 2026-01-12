import { useState, useRef, useEffect } from 'react'
import PropTypes from 'prop-types'

// Constants
const MAX_CHARS = 15000  // Support ~100 sentences for testing
const MIN_TEXTAREA_HEIGHT = 192
const MAX_TEXTAREA_HEIGHT = 600  // Taller for long text
const NEAR_LIMIT_THRESHOLD = 0.8
const CIRCLE_RADIUS = 10
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_RADIUS
const TRANSITION_DURATION_MS = 300
const TRANSITION_DURATION_LONG_MS = 500

function Editor({ value, onChange, placeholder, disabled, maxLength = MAX_CHARS }) {
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef(null)
  const charCount = value?.length || 0
  const isNearLimit = charCount > maxLength * NEAR_LIMIT_THRESHOLD
  const isAtLimit = charCount >= maxLength

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      const newHeight = Math.max(MIN_TEXTAREA_HEIGHT, Math.min(textareaRef.current.scrollHeight, MAX_TEXTAREA_HEIGHT))
      textareaRef.current.style.height = `${newHeight}px`
    }
  }, [value])

  const handleChange = (e) => {
    const newValue = e.target.value
    if (newValue.length <= maxLength) {
      onChange(newValue)
    }
  }

  return (
    <div className="relative group">
      {/* Animated border glow container */}
      <div
        className={`
          absolute -inset-[2px] rounded-2xl transition-all duration-500 ease-out
          ${isFocused
            ? 'bg-gradient-to-r from-nahawi-primary via-nahawi-secondary to-nahawi-accent opacity-100'
            : 'bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 opacity-0 group-hover:opacity-60'
          }
          ${disabled ? 'opacity-0' : ''}
        `}
        style={{
          filter: isFocused ? 'blur(4px)' : 'blur(2px)',
        }}
      />

      {/* Inner glow effect */}
      <div
        className={`
          absolute -inset-[1px] rounded-2xl transition-all duration-300
          ${isFocused
            ? 'bg-gradient-to-r from-nahawi-primary via-nahawi-secondary to-nahawi-accent'
            : 'bg-gray-200'
          }
          ${disabled ? 'bg-gray-100' : ''}
        `}
      />

      {/* Main textarea container */}
      <div className={`
        relative bg-white rounded-xl overflow-hidden
        transition-all duration-300 ease-out
        ${isFocused ? 'shadow-lg shadow-nahawi-secondary/20' : 'shadow-sm'}
        ${disabled ? 'bg-gray-50' : ''}
      `}>
        {/* Decorative corner accents */}
        <div className={`
          absolute top-0 right-0 w-16 h-16 pointer-events-none
          transition-opacity duration-300
          ${isFocused ? 'opacity-100' : 'opacity-0'}
        `}>
          <div className="absolute top-3 right-3 w-2 h-2 rounded-full bg-nahawi-accent/40" />
          <div className="absolute top-3 right-7 w-1 h-1 rounded-full bg-nahawi-secondary/30" />
          <div className="absolute top-7 right-3 w-1 h-1 rounded-full bg-nahawi-primary/30" />
        </div>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          id="arabic-text-input"
          value={value}
          onChange={handleChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={maxLength}
          dir="rtl"
          lang="ar"
          aria-label="النص العربي للتصحيح"
          aria-describedby="char-count-info writing-status"
          aria-invalid={isAtLimit}
          className={`
            w-full min-h-[192px] p-6 pb-10
            text-xl font-arabic leading-loose tracking-wide
            bg-transparent
            resize-none
            focus:outline-none
            transition-all duration-300
            arabic-textarea
            ${disabled
              ? 'text-gray-400 cursor-not-allowed'
              : 'text-gray-800'
            }
            placeholder:text-gray-300 placeholder:font-light
            placeholder:transition-all placeholder:duration-300
            focus:placeholder:text-gray-400 focus:placeholder:translate-x-1
            selection:bg-nahawi-accent/30
          `}
          style={{
            textRendering: 'optimizeLegibility',
            WebkitFontSmoothing: 'antialiased',
            MozOsxFontSmoothing: 'grayscale',
          }}
        />

        {/* Character counter */}
        <div
          id="char-count-info"
          className={`
          absolute bottom-3 left-4
          flex items-center gap-2
          text-sm font-medium
          transition-all duration-300
          ${isFocused ? 'opacity-100' : 'opacity-0 group-hover:opacity-70'}
        `}>
          {/* Progress ring */}
          <div className="relative w-6 h-6" aria-hidden="true">
            <svg className="w-6 h-6 transform -rotate-90">
              {/* Background ring */}
              <circle
                cx="12"
                cy="12"
                r="10"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-gray-200"
              />
              {/* Progress ring */}
              <circle
                cx="12"
                cy="12"
                r={CIRCLE_RADIUS}
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={`${(charCount / maxLength) * CIRCLE_CIRCUMFERENCE} ${CIRCLE_CIRCUMFERENCE}`}
                className={`
                  transition-all duration-300
                  ${isAtLimit
                    ? 'text-red-500'
                    : isNearLimit
                      ? 'text-amber-500'
                      : 'text-nahawi-secondary'
                  }
                `}
              />
            </svg>
          </div>

          {/* Character count text */}
          <span
            className={`
            tabular-nums
            ${isAtLimit
              ? 'text-red-500'
              : isNearLimit
                ? 'text-amber-500'
                : 'text-gray-400'
            }
          `}
            aria-live="polite"
          >
            <span className="sr-only">عدد الأحرف: </span>
            {charCount.toLocaleString('ar-EG')} / {maxLength.toLocaleString('ar-EG')}
            {isAtLimit && <span className="sr-only"> - تم الوصول إلى الحد الأقصى</span>}
            {isNearLimit && !isAtLimit && <span className="sr-only"> - اقتربت من الحد الأقصى</span>}
          </span>
        </div>

        {/* Writing mode indicator */}
        <div
          id="writing-status"
          className={`
          absolute bottom-3 right-4
          flex items-center gap-1.5
          transition-all duration-300
          ${isFocused ? 'opacity-100' : 'opacity-0'}
        `}>
          <div
            className={`
            w-1.5 h-1.5 rounded-full
            ${value?.length > 0 ? 'bg-emerald-400 animate-pulse' : 'bg-gray-300'}
          `}
            aria-hidden="true"
          />
          <span className="text-xs text-gray-400 font-arabic">
            {value?.length > 0 ? 'جاهز للمراجعة' : 'ابدأ الكتابة'}
          </span>
        </div>
      </div>

      {/* Subtle bottom shadow for depth */}
      <div className={`
        absolute -bottom-2 left-4 right-4 h-4
        bg-nahawi-secondary/5 blur-xl rounded-full
        transition-opacity duration-300
        ${isFocused ? 'opacity-100' : 'opacity-0'}
      `} />
    </div>
  )
}

Editor.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func.isRequired,
  placeholder: PropTypes.string,
  disabled: PropTypes.bool,
  maxLength: PropTypes.number,
}

Editor.defaultProps = {
  value: '',
  placeholder: '',
  disabled: false,
  maxLength: MAX_CHARS,
}

export default Editor
