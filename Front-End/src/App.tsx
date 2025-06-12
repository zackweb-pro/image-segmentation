import './App.css'
import ImageSegmenter from './components/ImageSegmenter'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Wall Segmentation</h1>
        <p className="app-description">
          Analyze and segment walls in architectural images with AI
        </p>
      </header>
      
      <main className="app-main">
        <ImageSegmenter />
      </main>
      
      <footer className="app-footer">
        <p>© 2023 Wall Segmentation Project</p>
      </footer>
    </div>
  )
}

export default App
