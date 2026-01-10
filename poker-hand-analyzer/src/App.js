import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React from 'react';
import PokerHandAnalyzer from './PokerHandAnalyzer';
import PushFoldGrid from './components/PushFoldGrid';
import EquityCalculator from './components/EquityCalculator';
import CardSelector from './components/CardSelector';
import ViewEmbeddings from './components/ViewEmbeddings';
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

            {/* Push-Fold Grid */}
            <Route
              path="/pushfold_grid"
              element={
                <div>
                  <PushFoldGrid />
                </div>
              }
            />

            {/* View Embeddings */}
            <Route
              path="/view_embeddings"
              element={
                <div>
                  <ViewEmbeddings />
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