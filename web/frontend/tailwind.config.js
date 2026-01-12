/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        arabic: ['Noto Sans Arabic', 'Tahoma', 'Arial', 'sans-serif'],
      },
      colors: {
        nahawi: {
          primary: '#1e40af',
          secondary: '#3b82f6',
          accent: '#60a5fa',
          error: {
            orthography: '#ef4444',
            spelling: '#f97316',
            morphology: '#3b82f6',
            syntax: '#8b5cf6',
            verb: '#06b6d4',
            article: '#10b981',
          }
        }
      }
    },
  },
  plugins: [],
}
