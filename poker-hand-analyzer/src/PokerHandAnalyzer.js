import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Mock data for demonstration - replace with your actual JSON data
const mockHandData = {
  "22": [0.00841001730973175, 0.008034581260230062, 0.08405726675911118, 0.00013576199430810016, 0.0226973905355108, 0.013930466015916642, 0.010071444233778301, 0.00023386792822728482, 0.009885412025393054, 0.004666670213220349, 0.00025187301400590376, 0.00024526063740019737, 0.0002464520748787839, 0.0002498536897450557, 0.0002523594362979127, 0.0002543558729511541, 0.0027824934173364775, 0.0066092206538769386, 0.0003304781169291446, 0.004083821300424882, 0.0014949120309417563, 0.001958012235776421, 0.002173796976741885, 0.0005969640405474478, 0.0005914892123320824, 0.0005922365584856025, 0.0005922365584856025, 0.0005930842555464459, 0.0005987759358117638, 0.0006020993001645586, 0.0006160277707598255, 0.001957340357132391, 0.012437471859103502, 0.021422187827684597, 0.0038365349896245992, 0.0026156481371312473, 0.003177389686145467, 0.00220166017416689, 0.0008132571644719924, 0.0001742287241943144, 0.00017448062425548405, 0.00017472965423116549, 0.00017636736122664324, 0.00018368829448971164, 0.00020370026789114405, 0.0003786340466569288, 0.09583551118152313, 0.049325713070374616, 0.023355532193726304, 0.08288152702396283, 0.02892262303769936, 0.02769536593973364, 0.017023871748309747, 0.007043591324757451, 0.00407518015777359, 0.00041585292411351675, 0.00029495558059067434, 0.0003335189410276212, 0.014707210583806267, 0.056164366492808177, 0.004568479175689469, 0.06681590496311583, 0.08334684647738504, 0.028287790607222, 0.010645969823190828, 0.015783723533855133, 0.00040328479482335, 0.0004441087482178445, 0.016680141100055404, 0.007869340020730147, 0.0314380429262547, 0.00784895693036568, 0.0070431286346795035, 0.0005471298572363542, 0.0005515113916336626, 0.0005690179985151272, 0.0010103286156502066, 0.017387111583431397, 0.007370851829990582, 0.0049869035565359085, 0.0007776299187582395, 0.0007776299187582395, 0.0012887771262151252, 0.001857225890835082, 0.0009493030239446788, 0.0011043364989134503, 0.0011246632984275567, 0.001654204241647626, 0.0011258216368682762, 0.0011775259274560768, 0.0013620180731890171, 0.0016690393946290458, 0.0017056807317018517, 0.0017087584074367635, 0.0018285219827627454, 0.002479801963801864, 0.002806961148531157, 0.002997142430798155, 0.004853774558581348, 0.007311790524685545],
  "AKs": Array.from({length: 100}, (_, i) => Math.random() * 0.1 * (100 - i) / 100),
  "AKo": Array.from({length: 100}, (_, i) => Math.random() * 0.08 * (100 - i) / 100),
  "QQ": Array.from({length: 100}, (_, i) => Math.random() * 0.12 * (100 - i) / 100),
  "AA": Array.from({length: 100}, (_, i) => Math.random() * 0.15 * (100 - i) / 100),
  "KK": Array.from({length: 100}, (_, i) => Math.random() * 0.13 * (100 - i) / 100),
};

// Create proper 13x13 grid - pairs on diagonal, suited above, offsuit below
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
const chartTypes = ['Rank Frequencies', 'Cumulative Rank Frequencies', 'Both Rank Frequencies and Cumulative Rank Frequencies'];
const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0', '#ffb347', '#87ceeb'];

const PokerHandAnalyzer = () => {
  const [selectedHands, setSelectedHands] = useState([]);
  const [displayMode, setDisplayMode] = useState('Compare');
  const [chartType, setChartType] = useState('Rank Frequencies');
  const [handData] = useState(mockHandData);

  // Force Range mode when "Both" is selected
  useEffect(() => {
    if (chartType === 'Both Rank Frequencies and Cumulative Rank Frequencies') {
      setDisplayMode('Range');
    }
  }, [chartType]);

  // Load actual hand data (replace with your data loading logic)
  useEffect(() => {
    // In a real app, load from /charts/hand_data.json
    // const loadData = async () => {
    //   try {
    //     const response = await fetch('/charts/hand_data.json');
    //     const data = await response.json();
    //     setHandData(data);
    //   } catch (error) {
    //     console.error('Error loading hand data:', error);
    //   }
    // };
    // loadData();
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

  const chartData = useMemo(() => {
    if (!selectedHands.length) return [];
    
    return Array.from({length: 100}, (_, i) => {
      const dataPoint = { bucket: i + 1 };
      selectedHands.forEach((hand) => {
        if (handData[hand]) {
          dataPoint[hand] = handData[hand][i] || 0;
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
          dataPoint[hand] = cumSum;
        }
      });
      return dataPoint;
    });
  }, [selectedHands, handData]);

  const renderCompareMode = () => {
    const hands = selectedHands.slice(0, 4);
    if (hands.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Select up to 4 hands to compare
        </div>
      );
    }

    const renderChart = (hand, idx) => {
      if (chartType === 'Rank Frequencies') {
        // Use pre-built chart images for Rank Frequencies
        return (
          <div className="chart-image-container">
            <img 
              src={`/charts/${hand}.png`} 
              alt={`${hand} distribution chart`}
              className="chart-image"
              onError={(e) => {
                // Fallback to generated chart if image not found
                console.log(`Chart image not found for ${hand}, using generated chart`);
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <ResponsiveContainer width="100%" height={200} style={{display: 'none'}}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Bar dataKey={hand} fill={colors[idx % colors.length]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      } else {
        // Use generated charts for Cumulative Rank Frequencies
        return (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={cumulativeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bucket" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey={hand} 
                stroke={colors[idx % colors.length]}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        );
      }
    };

    return (
      <div className="grid grid-cols-2 gap-4 h-full">
        {hands.map((hand, idx) => (
          <div key={hand} className="border rounded-lg p-4">
            <h3 className="text-lg font-bold text-center mb-2">{hand}</h3>
            {renderChart(hand, idx)}
          </div>
        ))}
        {/* Fill empty slots */}
        {Array.from({length: 4 - hands.length}).map((_, idx) => (
          <div key={`empty-${idx}`} className="border border-dashed border-gray-300 rounded-lg flex items-center justify-center text-gray-400">
            Empty Slot
          </div>
        ))}
      </div>
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

    const renderSingleChart = (data, title, type) => (
      <div className="h-full">
        <h3 className="text-xl font-bold text-center mb-2">{title}</h3>
        <ResponsiveContainer width="100%" height="90%">
          {type === 'bar' ? (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bucket" />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedHands.map((hand, idx) => (
                <Bar key={hand} dataKey={hand} fill={colors[idx % colors.length]} />
              ))}
            </BarChart>
          ) : (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bucket" />
              <YAxis />
              <Tooltip />
              <Legend />
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
          )}
        </ResponsiveContainer>
      </div>
    );

    if (chartType === 'Both Rank Frequencies and Cumulative Rank Frequencies') {
      return (
        <div className="h-full">
          <div className="text-center mb-4">
            <h2 className="text-xl font-bold">Range Analysis ({selectedHands.length} hands)</h2>
            <div className="text-sm text-gray-600">Selected: {selectedHands.join(', ')}</div>
          </div>
          <div className="grid grid-cols-1 gap-4" style={{height: 'calc(100% - 80px)'}}>
            <div className="h-1/2">
              <h3 className="text-lg font-bold text-center mb-2">Rank Frequencies</h3>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {selectedHands.map((hand, idx) => (
                    <Bar key={hand} dataKey={hand} fill={colors[idx % colors.length]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="h-1/2">
              <h3 className="text-lg font-bold text-center mb-2">Cumulative Rank Frequencies</h3>
              <ResponsiveContainer width="100%" height="90%">
                <LineChart data={cumulativeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
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
            </div>
          </div>
        </div>
      );
    }

    if (chartType === 'Rank Frequencies') {
      return (
        <div className="h-full">
          <div className="text-center mb-4">
            <h2 className="text-xl font-bold">Rank Frequencies ({selectedHands.length} hands)</h2>
            <div className="text-sm text-gray-600">Selected: {selectedHands.join(', ')}</div>
          </div>
          <div style={{height: 'calc(100% - 80px)'}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedHands.map((hand, idx) => (
                  <Bar key={hand} dataKey={hand} fill={colors[idx % colors.length]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      );
    }

    return (
      <div className="h-full">
        <div className="text-center mb-4">
          <h2 className="text-xl font-bold">{chartType} ({selectedHands.length} hands)</h2>
          <div className="text-sm text-gray-600">Selected: {selectedHands.join(', ')}</div>
        </div>
        <div style={{height: 'calc(100% - 80px)'}}>
          {chartType === 'Rank Frequencies' 
            ? renderSingleChart(chartData, '', 'bar')
            : renderSingleChart(cumulativeData, '', 'line')
          }
        </div>
      </div>
    );
  };

  const isDisplayModeDisabled = (mode) => {
    return chartType === 'Both Rank Frequencies and Cumulative Rank Frequencies' && mode === 'Compare';
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
                  disabled={isDisplayModeDisabled(mode)}
                  className={`mode-button ${
                    displayMode === mode 
                      ? 'active' 
                      : isDisplayModeDisabled(mode)
                      ? 'disabled'
                      : 'inactive'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Chart Type Selection */}
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