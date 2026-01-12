import { useEffect, useState } from 'react'
import PropTypes from 'prop-types'

// Constants
const ANIMATION_DELAY_MS = 100
const ORBIT_RADIUS_PX = 32
const DOT_PATTERN_SIZE_PX = 24
const STAGGER_BASE_DELAY_MS = 400
const STAGGER_INCREMENT_MS = 100

// Stats data configuration
const HEADER_STATS = [
  { id: 'f05', label: 'F0.5', value: '78.84%', icon: '/' },
  { id: 'errors', label: 'Error Types', labelAr: 'انواع الاخطاء', value: '18', icon: '!' },
  { id: 'sota', label: 'From SOTA', labelAr: 'من SOTA', value: '3.79', suffix: 'pts', icon: '^' },
]

function Header() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [hoveredStat, setHoveredStat] = useState(null)

  useEffect(() => {
    // Trigger animations after mount
    const timer = setTimeout(() => setIsLoaded(true), ANIMATION_DELAY_MS)
    return () => clearTimeout(timer)
  }, [])

  return (
    <header className="relative overflow-hidden" role="banner" aria-label="Nahawi - نظام تصحيح القواعد العربية">
      {/* Animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900" aria-hidden="true" />

      {/* Animated mesh gradient overlay */}
      <div className="absolute inset-0 opacity-60" aria-hidden="true">
        <div className="absolute top-0 -left-1/4 w-1/2 h-full bg-gradient-to-r from-violet-600/20 to-transparent blur-3xl animate-pulse animation-duration-4000" />
        <div className="absolute top-0 -right-1/4 w-1/2 h-full bg-gradient-to-l from-cyan-500/20 to-transparent blur-3xl animate-pulse animation-duration-5000 animation-delay-1000" />
        <div className="absolute -bottom-1/2 left-1/4 w-1/2 h-full bg-gradient-to-t from-indigo-600/20 to-transparent blur-3xl animate-pulse animation-duration-6000 animation-delay-2000" />
      </div>

      {/* Subtle dot pattern overlay */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        aria-hidden="true"
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, white 1px, transparent 1px)`,
          backgroundSize: `${DOT_PATTERN_SIZE_PX}px ${DOT_PATTERN_SIZE_PX}px`
        }}
      />

      {/* Glowing top border */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent" aria-hidden="true" />

      {/* Main content */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6">

          {/* Logo and title section */}
          <div
            className={`text-center sm:text-right transition-all duration-1000 ease-out ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            {/* Animated logo icon */}
            <div className="flex items-center justify-center sm:justify-end gap-4 mb-3">
              <div className="relative">
                {/* Outer glow ring */}
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 to-violet-500 blur-lg opacity-50 animate-pulse animation-duration-3000" />

                {/* Logo container */}
                <div className="relative w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-gradient-to-br from-cyan-400 via-indigo-500 to-violet-600 p-[2px]">
                  <div className="w-full h-full rounded-full bg-slate-900/90 flex items-center justify-center backdrop-blur-sm">
                    {/* Arabic letter Noon as logo */}
                    <span className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-cyan-300 to-violet-400 bg-clip-text text-transparent">
                      ن
                    </span>
                  </div>
                </div>

                {/* Orbiting dot */}
                <div
                  className="absolute w-2 h-2 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"
                  style={{
                    animation: 'orbit 8s linear infinite',
                    top: '50%',
                    left: '50%',
                  }}
                />
              </div>

              {/* Title */}
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold">
                <span className="bg-gradient-to-l from-white via-cyan-100 to-violet-200 bg-clip-text text-transparent drop-shadow-2xl">
                  نحوي
                </span>
              </h1>
            </div>

            {/* Tagline */}
            <p
              className={`text-sm sm:text-base text-slate-300/90 font-light tracking-wide transition-all duration-1000 delay-200 ${
                isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
              }`}
            >
              <span className="text-cyan-400/80">AI-Powered</span>
              <span className="mx-2 text-slate-500">|</span>
              تصحيح القواعد العربية بالذكاء الاصطناعي
            </p>
          </div>

          {/* Stats badges */}
          <div
            role="group"
            aria-label="إحصائيات أداء النموذج"
            className={`flex flex-wrap items-center justify-center gap-3 transition-all duration-1000 delay-300 ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            {/*
            Dynamic transition delays for staggered entrance animation.
            Uses inline style because delay varies by index (STAGGER_BASE_DELAY_MS + index * STAGGER_INCREMENT_MS).
            This creates a cascading reveal effect that cannot be achieved with static CSS classes.
          */}
          {HEADER_STATS.map((stat, index) => (
              <div
                key={stat.id}
                className="relative group cursor-default transition-all duration-500"
                style={{ transitionDelay: `${STAGGER_BASE_DELAY_MS + index * STAGGER_INCREMENT_MS}ms` }}
                onMouseEnter={() => setHoveredStat(stat.id)}
                onMouseLeave={() => setHoveredStat(null)}
              >
                {/* Glow effect on hover */}
                <div
                  className={`absolute -inset-1 rounded-2xl bg-gradient-to-r from-cyan-500/30 to-violet-500/30 blur-lg transition-opacity duration-300 ${
                    hoveredStat === stat.id ? 'opacity-100' : 'opacity-0'
                  }`}
                />

                {/* Glass card */}
                <div className={`
                  relative px-5 py-3 rounded-xl
                  bg-white/[0.05] backdrop-blur-md
                  border border-white/[0.08]
                  shadow-lg shadow-black/10
                  transition-all duration-300
                  hover:bg-white/[0.08] hover:border-white/[0.15]
                  hover:scale-105 hover:-translate-y-0.5
                `}>
                  {/* Shine effect */}
                  <div className="absolute inset-0 rounded-xl overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-br from-white/[0.08] via-transparent to-transparent" />
                  </div>

                  {/* Content */}
                  <div className="relative flex items-center gap-3">
                    {/* Stat icon */}
                    <div className={`
                      w-8 h-8 rounded-lg flex items-center justify-center text-sm font-mono
                      ${stat.id === 'f05' ? 'bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 text-cyan-300' : ''}
                      ${stat.id === 'errors' ? 'bg-gradient-to-br from-amber-500/20 to-orange-500/20 text-orange-300' : ''}
                      ${stat.id === 'sota' ? 'bg-gradient-to-br from-violet-500/20 to-pink-500/20 text-violet-300' : ''}
                    `}>
                      {stat.id === 'f05' && (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      )}
                      {stat.id === 'errors' && (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                        </svg>
                      )}
                      {stat.id === 'sota' && (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                        </svg>
                      )}
                    </div>

                    {/* Text content */}
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
                        {stat.label}
                      </span>
                      <div className="flex items-baseline gap-1">
                        <span className="text-lg font-bold text-white">
                          {stat.value}
                        </span>
                        {stat.suffix && (
                          <span className="text-xs text-slate-400">{stat.suffix}</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent" />

      {/* CSS for orbit animation */}
      <style>{`
        @keyframes orbit {
          from {
            transform: rotate(0deg) translateX(${ORBIT_RADIUS_PX}px) rotate(0deg);
          }
          to {
            transform: rotate(360deg) translateX(${ORBIT_RADIUS_PX}px) rotate(-360deg);
          }
        }
      `}</style>
    </header>
  )
}

// Header component has no props currently, but PropTypes included for future extensibility
Header.propTypes = {}

Header.defaultProps = {}

export default Header
