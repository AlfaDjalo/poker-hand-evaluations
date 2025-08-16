import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Create 13x13 starting hand grid - pairs on diagonal, suited above, offsuit below
const createHandGrid = () => {
  const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
  const grid = [];
  
  for (let row = 0; row < 13; row++) {
    const gridRow = [];
    for (let col = 0; col < 13; col++) {
      if (row === col) {
        // Pairs on diagonal
        gridRow.push(ranks[row] + ranks[col]);
      } else if (row < col) {
        // Suited hands above diagonal - higher rank first
        gridRow.push(ranks[row] + ranks[col] + 's');
      } else {
        // Offsuit hands below diagonal - higher rank first
        gridRow.push(ranks[col] + ranks[row] + 'o');
      }
    }
    grid.push(gridRow);
  }
  
  return grid;
};

const HAND_GRID = createHandGrid();

const displayModes = ['Compare', 'Range'];
const chartTypes = ['Rank Frequencies', 'Cumulative Rank Frequencies'];
const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0', '#ffb347', '#87ceeb'];

const PokerHandAnalyzer = () => {
  const [selectedHands, setSelectedHands] = useState([]);
  const [displayMode, setDisplayMode] = useState('Compare');
  const [chartType, setChartType] = useState('Rank Frequencies');
  const [handData, setHandData] = useState({});
  const [dataLoaded, setDataLoaded] = useState(false);
  const [dataError, setDataError] = useState(null);

  // Load actual hand data from JSON file
  useEffect(() => {
    const loadData = async () => {
      try {
        setDataError(null);
        const response = await fetch('/charts/chart_data.json');
        if (!response.ok) {
          throw new Error(`Failed to load chart data: ${response.status}`);
        }
        const data = await response.json();
        setHandData(data);
        setDataLoaded(true);
        console.log('Chart data loaded successfully', Object.keys(data).length, 'hands');
      } catch (error) {
        console.error('Error loading hand data:', error);
        setDataError(error.message);
      }
    };
    loadData();
  }, []);

  const toggleHandSelection = (hand) => {
    setSelectedHands(prev => {
      if (prev.includes(hand)) {
        return prev.filter(h => h !== hand);
      } else {
        if (displayMode === 'Compare' && prev.length >= 4) {
          return prev; // Limit to 4 hands in compare mode
        }
        return [...prev, hand];
      }
    });
  };

  // Handle switching from Range to Compare mode - limit to 4 hands
    useEffect(() => {
      if (displayMode === 'Compare' && selectedHands.length > 4) {
        setSelectedHands(prev => prev.slice(0, 4));
      }
    }, [displayMode, selectedHands.length]);

  const getHandStyle = (hand, row, col) => {
    const isPair = row === col;
    const isSuited = row < col;
    const isOffsuit = row > col;
    
    let baseColor = '#dc2626'; // Red for pairs
    if (isPair) baseColor = '#dc2626'; // Red for pairs
    else if (isSuited) baseColor = '#16a34a'; // Green for suited
    else if (isOffsuit) baseColor = '#2563eb'; // Blue for offsuit
    
    return {
      backgroundColor: baseColor,
      opacity: 0.8,
    };
  };

  // Custom tick formatter for percentages
  const formatPercent = (value) => `${value}%`;
  
  // Custom tick formatter for x-axis (showing every 10%)
  const formatXPercent = (value) => value % 10 === 0 ? `${value}%` : '';

  const chartData = useMemo(() => {
    if (!selectedHands.length) return [];
    
    return Array.from({length: 100}, (_, i) => {
      const dataPoint = { bucket: i + 1 };
      selectedHands.forEach((hand) => {
        if (handData[hand]) {
          dataPoint[hand] = (handData[hand][i] || 0) * 100;
        }
      });
      return dataPoint;
    });
  }, [selectedHands, handData]);

  const cumulativeData = useMemo(() => {
    if (!selectedHands.length) return [];
    
    return Array.from({length: 100}, (_, i) => {
      const dataPoint = { bucket: i + 1 };
      selectedHands.forEach(hand => {
        if (handData[hand]) {
          const cumSum = handData[hand].slice(0, i + 1).reduce((sum, val) => sum + val, 0);
          dataPoint[hand] = cumSum * 100;
        }
      });
      return dataPoint;
    });
  }, [selectedHands, handData]);

  // Range data - average of selected hands
  const rangeData = useMemo(() => {
    if (!selectedHands.length || !dataLoaded) return [];
    
    return Array.from({length: 100}, (_, i) => {
      let sum = 0;
      let validHands = 0;
      
      selectedHands.forEach(hand => {
        if (handData[hand] && handData[hand][i] !== undefined) {
          sum += handData[hand][i];
          validHands++;
        }
      });

      return {
        bucket: i + 1,
        Range: validHands > 0 ? (sum / validHands) * 100 : 0 // Average and convert to percentage
      };
    });
  }, [selectedHands, handData, dataLoaded]);

  const rangeCumulativeData = useMemo(() => {
    if (!selectedHands.length || !dataLoaded) return [];
    
    return Array.from({length: 100}, (_, i) => {
      let sum = 0;
      let validHands = 0;
      
      selectedHands.forEach(hand => {
        if (handData[hand]) {
          const cumSum = handData[hand].slice(0, i + 1).reduce((s, val) => s + (val || 0), 0);
          sum += cumSum;
          validHands++;
        }
      });
      
      return {
        bucket: i + 1,
        Range: validHands > 0 ? (sum / validHands) * 100 : 0 // Average and convert to percentage
      };
    });
  }, [selectedHands, handData, dataLoaded]);

  const renderCompareMode = () => {
    const hands = selectedHands.slice(0, 4);
    if (hands.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Select up to 4 hands to compare
        </div>
      );
    }

    console.log(chartData)

    const renderBarChart = (hand, idx) => {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="bucket" 
              tickFormatter={formatXPercent}
              domain={[1, 100]}
              ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
              label={{ value: 'Hand rank percentile', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              tickFormatter={formatPercent}
              domain={[0, 20]}
              label={{ value: 'Frequency (normalized)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Bar dataKey={hand} fill={colors[idx % colors.length]} />
          </BarChart>
        </ResponsiveContainer>
      );
    };

    const renderLineChart = (data) => {
      return (
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="bucket" 
              tickFormatter={formatXPercent}
              domain={[1, 100]}
              ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
              label={{ value: 'Hand rank percentile', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              tickFormatter={formatPercent}
              domain={[1, 100]}
              ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
              label={{ value: 'Frequency (normalized)', angle: -90, position: 'insideLeft' }}
            />
            <Legend/>
            <Tooltip />
            {selectedHands.map((hand, idx) => (
              <Line 
                key={hand}
                type="monotone" 
                dataKey={hand} 
                stroke={colors[idx % colors.length]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    };

    return (
      <>
        {hands.length === 1 ? (
          <div className="flex flex-col h-full">
            <div className="h-1/2 p-4">
              <h3 className="text-lg font-bold text-center mb-2">{hands[0]}</h3>
              {renderBarChart(hands[0], 0)}
            </div>
            <div className="h-full w-full border rounded-lg p-4">
            {/* <div className="h-1/2 p-4"> */}
              <h3 className="text-lg font-bold text-center mb-2">Cumulative Rank Frequencies</h3>
              {renderLineChart(cumulativeData)}
            </div>
          </div>
        ) : chartType === 'Rank Frequencies' ? (
          <div className="grid grid-cols-2 gap-4 h-full">
            {hands.map((hand, idx) => (
              <div key={hand} className="border rounded-lg p-4">
                <h3 className="text-lg font-bold text-center mb-2">{hand}</h3>
                {renderBarChart(hand, idx)}
              </div>
            ))}
            {Array.from({length: 4 - hands.length}).map((_, idx) => (
              <div key={`empty-${idx}`} className="border border-dashed border-gray-300 rounded-lg flex items-center justify-center text-gray-400">
                {/* Empty Slot */}
              </div>
            ))}
          </div>
        ) : (
          <div className="h-full w-full border rounded-lg p-4">
            <h3 className="text-lg font-bold text-center mb-2">Cumulative Rank Frequencies</h3>
            {renderLineChart(cumulativeData)}
          </div>
        )}
      </>
    );
  };

  const renderRangeMode = () => {
    if (selectedHands.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Select hands to view range distribution
        </div>
      );
    }

    if (!dataLoaded) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Loading chart data...
        </div>
      );
    }

    if (dataError) {
      return (
        <div className="flex items-center justify-center h-full text-red-500">
          Error loading data: {dataError}
        </div>
      );
    }

    const renderSingleChart = (data, title, type) => (
      <div className="h-full">
        <h3 className="text-xl font-bold text-center mb-2">{title}</h3>
        <ResponsiveContainer width="100%" height="90%">
          {type === 'bar' ? (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="bucket" 
                tickFormatter={formatXPercent}
                domain={[1, 100]}
                ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
                label={{ value: 'Hand rank percentile', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                tickFormatter={formatPercent}
                domain={[0, 20]}
                label={{ value: 'Frequency (normalized)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              {/* <Legend /> */}
              <Bar key="Range" dataKey="Range" fill={colors[0]} />
              {/* <Bar key="Range" dataKey="Range" fill={colors[0]} /> */}
              {/* {selectedHands.map((hand, idx) => (
                <Bar key={hand} dataKey={hand} fill={colors[idx % colors.length]} />
              ))} */}
            </BarChart>
          ) : (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="bucket" 
                tickFormatter={formatXPercent}
                domain={[1, 100]}
                ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
                label={{ value: 'Hand rank percentile', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                tickFormatter={formatPercent}
                domain={[1, 100]}
                ticks={[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
                label={{ value: 'Frequency (normalized)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              {/* <Legend /> */}
              {selectedHands.map((hand, idx) => (
                <Line 
                  key="Range"
                  type="monotone" 
                  dataKey="Range" 
                  stroke={colors[0]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    );

    return (
      <div className="h-full">
        <div className="text-center mb-4">
          <h2 className="text-xl font-bold">{chartType} for selected Range</h2>
        </div>
        <div className="flex flex-col gap-4" style={{ height: 'calc(100% - 80px)' }}>
          <div style={{ height: '33%' }}>
            {renderSingleChart(rangeData, 'Rank Frequencies', 'bar')}
            {/* {renderSingleChart(chartData, 'Rank Frequencies', 'bar')} */}
          </div>
          <div style={{ height: '67%' }}>
            {renderSingleChart(rangeCumulativeData, 'Cumulative Rank Frequencies', 'line')}
            {/* {renderSingleChart(cumulativeData, 'Cumulative Rank Frequencies', 'line')} */}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Left Panel - Hand Grid */}
      <div className="left-panel">
        <div>
          <h2 className="text-xl font-bold mb-4">Texas Hold'em Starting Hands</h2>
          
          {/* Display Mode Selection */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Display Mode:</h3>
            <div className="flex gap-2">
              {displayModes.map(mode => (
                <button
                  key={mode}
                  onClick={() => setDisplayMode(mode)}
                  className={`mode-button ${
                    displayMode === mode 
                      ? 'active' 
                      : 'inactive'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {displayMode === 'Compare' && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold mb-2">Chart Type:</h3>
              <div className="flex flex-col gap-2">
                {chartTypes.map(type => (
                  <label key={type} className="flex items-center">
                    <input
                      type="radio"
                      name="chartType"
                      value={type}
                      checked={chartType === type}
                      onChange={(e) => setChartType(e.target.value)}
                      style={{marginRight: '8px'}}
                    />
                    <span className="text-sm">{type}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="text-sm text-gray-600 mb-2">
            Selected: {selectedHands.length} hand{selectedHands.length !== 1 ? 's' : ''}
            {displayMode === 'Compare' && ' (max 4)'}
          </div>
          
          {selectedHands.length > 0 && (
            <button
              onClick={() => setSelectedHands([])}
              className="text-sm text-red-600 mb-4"
              style={{textDecoration: 'underline', background: 'none', border: 'none', cursor: 'pointer'}}
            >
              Clear All
            </button>
          )}
        </div>
        
        {/* 13x13 Hand Grid */}
        <div className="hand-grid">
          {HAND_GRID.map((row, rowIdx) => 
            row.map((hand, colIdx) => (
              <div
                key={`${rowIdx}-${colIdx}`}
                className={`hand-cell ${selectedHands.includes(hand) ? 'selected' : ''}`}
                style={getHandStyle(hand, rowIdx, colIdx)}
                onClick={() => toggleHandSelection(hand)}
              >
                {hand}
              </div>
            ))
          )}
        </div>
        

      </div>

      {/* Right Panel - Charts */}
      <div className="right-panel">
        <div className="chart-container">
          {displayMode === 'Compare' && renderCompareMode()}
          {displayMode === 'Range' && renderRangeMode()}
        </div>
      </div>
    </div>
  );
};

export default PokerHandAnalyzer;