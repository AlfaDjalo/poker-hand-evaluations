import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React from 'react';
import PokerHandAnalyzer from './PokerHandAnalyzer';
import EquityCalculator from './components/EquityCalculator';
import CardSelector from './components/CardSelector';
import GameSimulator from './components/GameSimulator';
import { Navbar } from "./components/Navbar";
import './App.css';

function App() {
  return (
    <Router>
      {/* Always visible */}
      <Navbar />

        <div className="pt-16">
          <Routes>

            {/* Equity Calculator */}
            <Route
              path="/"
              element={
                <div>
                  <CardSelector />
                </div>
              }
            />

            {/* Rank Chart */}
            <Route
              path="/rank_chart"
              element={
                <div>
                  <PokerHandAnalyzer />
                </div>
              }
            />

            {/* Equity Calculator */}
            <Route
              path="/equity_calculator"
              element={
                <div>
                  <EquityCalculator />
                </div>
              }
            />

            {/* Game Simulator */}
            <Route
              path="/game_simulator"
              element={
                <div>
                  <GameSimulator />
                </div>
              }
            />
          </Routes>
        </div >
    </Router>
  );
}

export default App;