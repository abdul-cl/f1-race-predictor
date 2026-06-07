import { useEffect, useState } from "react";
import "../App.css";
export default function CustomPredictor() {
  const [track, setTrack] = useState("Monaco");
  const [drivers, setDrivers] = useState([]);
  const [grid, setGrid] = useState({});
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadDrivers() {
      const response = await fetch("http://127.0.0.1:8000/drivers");
      const data = await response.json();

      setDrivers(data);

      const defaultGrid = {};
      data.forEach((driver, index) => {
        defaultGrid[driver.Driver] = index + 1;
      });

      setGrid(defaultGrid);
    }

    loadDrivers();
  }, []);

  function updateGrid(driverName, value) {
    setGrid({
      ...grid,
      [driverName]: Number(value),
    });
  }

  async function getPredictions() {
    setError("");
    setPredictions([]);

    const gridOrder = drivers.map((driver) => ({
      driver: driver.Driver,
      grid: grid[driver.Driver],
    }));

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          track: track,
          grid_order: gridOrder,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail);
        return;
      }

      setPredictions(data);
    } catch {
      setError("Could not connect to backend API.");
    }
  }

  return (
    <div className="app">
      <h1>F1 Race Predictor</h1>

      <div className="track-section">
  <input
    value={track}
    onChange={(e) => setTrack(e.target.value)}
    placeholder="Example: Monaco"
  />
</div>

<h2>Custom Grid Positions</h2>

      <div className="grid-list">
        {drivers.map((driver) => (
          <div className="grid-row" key={driver.Driver}>
            <span>{driver.Driver}</span>
            <input
              type="number"
              min="1"
              value={grid[driver.Driver] || ""}
              onChange={(e) => updateGrid(driver.Driver, e.target.value)}
            />
          </div>
        ))}
      </div>

      <button onClick={getPredictions}>Predict Race</button>

      {error && <p className="error">{error}</p>}

      {predictions.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Driver</th>
              <th>Team</th>
              <th>Grid</th>
              <th>Win Chance</th>
            </tr>
          </thead>

          <tbody>
            {predictions.map((driver, index) => (
              <tr key={index}>
                <td>{driver.Driver}</td>
                <td>{driver.Team}</td>
                <td>{driver["Starting Grid"]}</td>
                <td>{(driver.win_probability * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
