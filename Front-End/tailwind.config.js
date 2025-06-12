/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3498db',
          dark: '#2980b9',
        },
        accent: '#e74c3c',
        gray: {
          light: '#ecf0f1',
          DEFAULT: '#7f8c8d',
        },
        textColor: '#2c3e50',
      },
    },
  },
  plugins: [],
}
