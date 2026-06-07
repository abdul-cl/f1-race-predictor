import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import Home from "./pages/Home";
import CustomPredictor from "./pages/CustomPredictor";
import ChampionshipPrediction from "./pages/ChampionshipPrediction";
import QualiPrediction from "./pages/QualiPrediction";

import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <div className="app">

        <h1>F1 Analytics Platform</h1>

        <nav className="navbar">
          <Link to="/">Home</Link>

          <Link to="/custom-predictor">
            Custom Predictor
          </Link>

          <Link to="/championship-prediction">
            Championship Prediction
          </Link>

          <Link to="/quali-prediction">
            Qualifying Prediction
          </Link>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />

          <Route
            path="/custom-predictor"
            element={<CustomPredictor />}
          />

          <Route
            path="/championship-prediction"
            element={<ChampionshipPrediction />}
          />

          <Route
            path="/quali-prediction"
            element={<QualiPrediction />}
          />
        </Routes>

      </div>
    </BrowserRouter>
  );
}

export default App;