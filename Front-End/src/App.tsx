import ImageSegmenter from './components/ImageSegmenter'

function App() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <header className="bg-gradient-to-r from-[#2980b9] to-[#3498db] py-8 px-4 shadow-lg">
        <div className="container mx-auto text-center">
          <h1 className="text-4xl font-bold text-white tracking-tight mb-2">
            AI Wall Segmentation
          </h1>
          <p className="text-white text-opacity-90 max-w-2xl mx-auto text-lg">
            Advanced machine learning to automatically detect and highlight walls in your architectural images
          </p>
        </div>
      </header>
      
      <main className="flex-1 container mx-auto px-4 py-10 max-w-6xl">
        <ImageSegmenter />
      </main>
      
      <footer className="bg-gradient-to-r from-[#2c3e50] to-[#34495e] text-white py-6 px-4 mt-auto">
        <div className="container mx-auto flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm mb-3 md:mb-0">© {new Date().getFullYear()} Wall Segmentation Project</p>
          <div className="flex space-x-4">
            <a href="#" className="text-white hover:text-[#3498db] transition-colors">Documentation</a>
            <a href="#" className="text-white hover:text-[#3498db] transition-colors">GitHub</a>
            <a href="#" className="text-white hover:text-[#3498db] transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
