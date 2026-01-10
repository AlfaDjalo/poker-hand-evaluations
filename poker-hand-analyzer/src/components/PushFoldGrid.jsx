import React, { useState, useEffect } from 'react';
import { fetchPushFoldGrid } from '../utils/api';
import ComboSuitGrid from './ComboSuitGrid';
import "../css/PushFoldGrid.css";

// Create 13x13 starting hand grid
const createHandGrid = () => {
  const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
  const grid = [];
  
  for (let row = 0; row < 13; row++) {
    const gridRow = [];
    for (let col = 0; col < 13; col++) {
      if (row === col) {
        gridRow.push(ranks[row] + ranks[col]);
      } else if (row < col) {
        gridRow.push(ranks[row] + ranks[col] + 's');
      } else {
        gridRow.push(ranks[col] + ranks[row] + 'o');
      }
    }
    grid.push(gridRow);
  }
  
  return grid;
};

const HAND_GRID = createHandGrid();

const PushFoldGrid = () => {
  const [stackSize, setStackSize] = useState(10);
  const [position, setPosition] = useState('sb');
  const [mode, setMode] = useState('probs');
  const [gridData, setGridData] = useState(null);
  const [selectedHand, setSelectedHand] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadGridData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await fetchPushFoldGrid({
          stackBB: stackSize,
          position: position,
          mode: mode
        });
        setGridData(data);
      } catch (err) {
        console.error('Error loading grid data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    loadGridData();
  }, [stackSize, position, mode]);

  const getHandStyle = (hand, row, col, value) => {
    let backgroundColor;

    if (mode === 'probs') {
        // Probabilities (0 to 1 scale)
        if (value >= 0.75) {
        backgroundColor = '#16a34a'; // Dark green
        } else if (value >= 0.5) {
        backgroundColor = '#22c55e'; // Green
        } else if (value >= 0.25) {
        backgroundColor = '#eab308'; // Yellow
        } else {
        backgroundColor = '#dc2626'; // Red
        }
    } else if (mode === 'values') {
        // Values (EV) - simple green for positive, red for negative
        if (value > 0) {
        backgroundColor = '#16a34a'; // Green
        } else if (value < 0) {
        backgroundColor = '#dc2626'; // Red
        } else {
        backgroundColor = '#eab308'; // Yellow-ish for zero or neutral
        }
    }

    return {
        backgroundColor,
        opacity: selectedHand === hand ? 1.0 : 0.85,
        border: selectedHand === hand ? '3px solid white' : 'none',
    };
    };

  const getHandStyleOld = (hand, row, col, probability) => {
    // const isPair = row === col;
    // const isSuited = row < col;
    
    // Color based on probability (green = high, red = low)
    let backgroundColor;
    if (probability >= 0.75) {
      backgroundColor = '#16a34a'; // Dark green
    } else if (probability >= 0.5) {
      backgroundColor = '#22c55e'; // Green
    } else if (probability >= 0.25) {
      backgroundColor = '#eab308'; // Yellow
    } else {
      backgroundColor = '#dc2626'; // Red
    }
    
    return {
      backgroundColor,
      opacity: selectedHand === hand ? 1.0 : 0.85,
      border: selectedHand === hand ? '3px solid white' : 'none',
    };
  };

    const getCellValue = (row, col) => {
    if (!gridData) return 0;

    if (mode === 'probs') {
        return gridData.prob_grid?.[row]?.[col] ?? 0;
    } else if (mode === 'values') {
        return gridData.value_grid?.[row]?.[col] ?? 0;
    }
    return 0;
    };

  const renderComboBreakdown = () => {
    if (!selectedHand || !gridData || !gridData.combos) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Select a hand to view combo breakdown
        </div>
      );
    }

    const handData = gridData.combos[selectedHand];
    if (!handData) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          No data for selected hand
        </div>
      );
    }


    return (
      <div className="p-6">
        <ComboSuitGrid
        combos={handData.combos}
        mode={mode}
        />
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Left Panel - Controls and Grid */}
      <div className="left-panel">
        <div>
          <h2 className="text-xl font-bold mb-4">Push-Fold Strategy</h2>
          
          {/* Stack Size Input */}
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">
              Stack Size (BB): {stackSize}
            </label>
            <input
              type="range"
              min="2"
              max="20"
              step="0.5"
              value={stackSize}
              onChange={(e) => setStackSize(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>2bb</span>
              <span>20bb</span>
            </div>
          </div>

          {/* Position Selection */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Position:</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setPosition('sb')}
                className={`mode-button ${
                  position === 'sb' ? 'active' : 'inactive'
                }`}
              >
                SB (Push)
              </button>
              <button
                onClick={() => setPosition('bb')}
                className={`mode-button ${
                  position === 'bb' ? 'active' : 'inactive'
                }`}
              >
                BB (Call)
              </button>
            </div>
          </div>

          {/* Mode Selection */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Display:</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setMode('probs')}
                className={`mode-button ${
                  mode === 'probs' ? 'active' : 'inactive'
                }`}
              >
                Probabilities
              </button>
              <button
                onClick={() => setMode('values')}
                className={`mode-button ${
                  mode === 'values' ? 'active' : 'inactive'
                }`}
              >
                Values
              </button>
            </div>
          </div>

          {/* Legend
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Color Legend:</h3>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-600"></div>
                <span>75-100% (Strong {position === 'sb' ? 'Push' : 'Call'})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500"></div>
                <span>50-75% (Medium)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-yellow-500"></div>
                <span>25-50% (Weak)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-600"></div>
                <span>0-25% (Fold)</span>
              </div>
            </div>
          </div> */}

          {/* {selectedHand && (
            <div className="text-sm text-gray-600 mb-2">
              Selected: <span className="font-bold">{selectedHand}</span>
            </div>
          )} */}
        </div>
        
        {/* Hand Grid */}
        {loading ? (
        <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Loading grid...</div>
        </div>
        ) : error ? (
        <div className="flex items-center justify-center h-64">
            <div className="text-red-500">Error: {error}</div>
        </div>
        ) : (
        <div className="hand-grid">
            {HAND_GRID.map((row, rowIdx) =>
            row.map((hand, colIdx) => {
                const value = getCellValue(rowIdx, colIdx);
                return (
                <div
                    key={`${rowIdx}-${colIdx}`}
                    className={`hand-cell ${selectedHand === hand ? 'selected' : ''}`}
                    style={getHandStyle(hand, rowIdx, colIdx, value)}
                    onClick={() => setSelectedHand(hand)}
                >
                    <div className="font-bold text-sm">{hand}</div>
                    <div className="text-xs mt-1">
                    {mode === 'probs' ? `${(value * 100).toFixed(0)}%` : value.toFixed(2)}
                    </div>
                </div>
                );
            })
            )}
        </div>
        )}
      </div>

      {/* Right Panel - Combo Breakdown */}
      <div className="right-panel">
        {selectedHand && (
        <div className="text-sm text-gray-600 mb-2">
            Selected: <span className="font-bold">{selectedHand}</span>
        </div>
        )}
        <div className="chart-container">
          {renderComboBreakdown()}
        </div>
      </div>
    </div>
  );
};

export default PushFoldGrid;