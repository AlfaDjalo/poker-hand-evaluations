import React, { useState, useCallback } from 'react';
import { ChevronDown, Calculator, Shuffle } from 'lucide-react';

const SUITS = ['♠', '♥', '♦', '♣'];
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

const VARIANTS = [
  { value: 'plo', label: 'PLO High' },
  { value: 'plo8', label: 'PLO Eight-or-Better Hi/Lo' },
  { value: 'plo5', label: 'PLO 5-Card' },
  { value: 'plo6', label: 'PLO 6-Card' }
];

// Generate full deck
const generateDeck = () => {
  const deck = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ rank, suit, id: `${rank}${suit}` });
    }
  }
  return deck;
};

const Card = ({ card, onClick, isSelected, isClickable = true, size = 'normal' }) => {
  const isRed = card?.suit === '♥' || card?.suit === '♦';
  
  const sizeClasses = {
    small: 'w-8 h-12 text-xs',
    normal: 'w-12 h-16 text-sm',
    large: 'w-16 h-20 text-base'
  };

  if (!card) {
    return (
      <div className={`${sizeClasses[size]} bg-blue-800 border-2 border-white rounded-lg shadow-lg flex items-center justify-center cursor-pointer hover:bg-blue-700 transition-colors`}>
        <div className="w-full h-full bg-blue-600 rounded-md opacity-50"></div>
      </div>
    );
  }

  return (
    <div 
      className={`${sizeClasses[size]} bg-white border-2 rounded-lg shadow-lg flex flex-col items-center justify-between p-1 transition-all duration-200 ${
        isClickable ? 'cursor-pointer hover:shadow-xl hover:-translate-y-1' : ''
      } ${
        isSelected ? 'border-yellow-400 bg-yellow-50' : 'border-gray-300'
      }`}
      onClick={isClickable ? onClick : undefined}
    >
      <div className={`font-bold ${isRed ? 'text-red-600' : 'text-black'}`}>
        {card.rank}
      </div>
      <div className={`text-lg ${isRed ? 'text-red-600' : 'text-black'}`}>
        {card.suit}
      </div>
      <div className={`font-bold ${isRed ? 'text-red-600' : 'text-black'} transform rotate-180`}>
        {card.rank}
      </div>
    </div>
  );
};

const PlayerPosition = ({ position, cards, onCardClick, selectedCards, label }) => {
  const positionStyles = {
    top: 'absolute top-4 left-1/2 transform -translate-x-1/2',
    topRight: 'absolute top-8 right-8',
    right: 'absolute top-1/2 right-4 transform -translate-y-1/2',
    bottomRight: 'absolute bottom-8 right-8',
    bottom: 'absolute bottom-4 left-1/2 transform -translate-x-1/2',
    bottomLeft: 'absolute bottom-8 left-8',
    left: 'absolute top-1/2 left-4 transform -translate-y-1/2',
    topLeft: 'absolute top-8 left-8',
  };

  return (
    <div className={positionStyles[position]}>
      <div className="text-white text-sm mb-1 text-center">{label}</div>
      <div className="flex gap-1">
        {cards.map((card, index) => (
          <Card
            key={index}
            card={card}
            onClick={() => onCardClick && onCardClick(position, index)}
            isSelected={selectedCards?.includes(`${position}-${index}`)}
            size="small"
          />
        ))}
      </div>
    </div>
  );
};

const PokerTable = () => {
  const [variant, setVariant] = useState('plo');
  const [selectedCards, setSelectedCards] = useState([]);
  const [availableCards, setAvailableCards] = useState(generateDeck());
  const [isCalculating, setIsCalculating] = useState(false);
  
  // Player positions with 4 cards each (empty initially)
  const [playerHands, setPlayerHands] = useState({
    top: [null, null, null, null],
    topRight: [null, null, null, null],
    right: [null, null, null, null],
    bottomRight: [null, null, null, null],
    bottom: [null, null, null, null],
    bottomLeft: [null, null, null, null],
    left: [null, null, null, null],
    topLeft: [null, null, null, null],
  });

  // Board cards (flop, turn, river)
  const [board, setBoard] = useState([null, null, null, null, null]);
  
  // Dead cards
  const [deadCards, setDeadCards] = useState([]);

  const handleCardSelection = (card) => {
    // Toggle card selection from the deck
    const cardId = card.id;
    if (selectedCards.includes(cardId)) {
      setSelectedCards(selectedCards.filter(id => id !== cardId));
    } else {
      setSelectedCards([...selectedCards, cardId]);
    }
  };

  const handlePositionCardClick = (position, cardIndex) => {
    // Remove card from position and return it to available cards
    const currentCard = playerHands[position][cardIndex];
    if (currentCard) {
      const newHands = { ...playerHands };
      newHands[position] = [...newHands[position]];
      newHands[position][cardIndex] = null;
      setPlayerHands(newHands);
      
      // Add card back to available deck
      setAvailableCards([...availableCards, currentCard]);
    }
  };

  const handleBoardCardClick = (cardIndex) => {
    // Remove card from board and return it to available cards
    const currentCard = board[cardIndex];
    if (currentCard) {
      const newBoard = [...board];
      newBoard[cardIndex] = null;
      setBoard(newBoard);
      
      // Add card back to available deck
      setAvailableCards([...availableCards, currentCard]);
    }
  };

  const dealSelectedCards = () => {
    if (selectedCards.length === 0) return;
    
    // Find first available position for each selected card
    const selectedCardObjs = selectedCards.map(cardId => 
      availableCards.find(card => card.id === cardId)
    ).filter(Boolean);

    let cardIndex = 0;
    const newHands = { ...playerHands };
    const newBoard = [...board];
    
    // Try to fill player hands first
    for (const position of Object.keys(playerHands)) {
      for (let i = 0; i < 4 && cardIndex < selectedCardObjs.length; i++) {
        if (newHands[position][i] === null) {
          newHands[position][i] = selectedCardObjs[cardIndex];
          cardIndex++;
        }
      }
    }
    
    // Then fill board
    for (let i = 0; i < 5 && cardIndex < selectedCardObjs.length; i++) {
      if (newBoard[i] === null) {
        newBoard[i] = selectedCardObjs[cardIndex];
        cardIndex++;
      }
    }
    
    setPlayerHands(newHands);
    setBoard(newBoard);
    
    // Remove dealt cards from available deck
    const newAvailableCards = availableCards.filter(card => 
      !selectedCardObjs.includes(card)
    );
    setAvailableCards(newAvailableCards);
    setSelectedCards([]);
  };

  const shuffleAndDeal = () => {
    // Reset everything
    setPlayerHands({
      top: [null, null, null, null],
      topRight: [null, null, null, null],
      right: [null, null, null, null],
      bottomRight: [null, null, null, null],
      bottom: [null, null, null, null],
      bottomLeft: [null, null, null, null],
      left: [null, null, null, null],
      topLeft: [null, null, null, null],
    });
    setBoard([null, null, null, null, null]);
    setDeadCards([]);
    setAvailableCards(generateDeck());
    setSelectedCards([]);
  };

  const calculateEquity = async () => {
    setIsCalculating(true);
    
    try {
      // Collect all active hands
      const activeHands = Object.values(playerHands)
        .filter(hand => hand.some(card => card !== null))
        .map(hand => hand.filter(card => card !== null).map(card => card.rank + card.suit.replace('♠', 's').replace('♥', 'h').replace('♦', 'd').replace('♣', 'c')));
      
      // Collect board cards
      const boardCards = board
        .filter(card => card !== null)
        .map(card => card.rank + card.suit.replace('♠', 's').replace('♥', 'h').replace('♦', 'd').replace('♣', 'c'));
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock response
      const mockResults = activeHands.map((_, index) => (Math.random() * 0.8 + 0.1).toFixed(3));
      
      alert(`Equity Results:\n${activeHands.map((hand, index) => 
        `Player ${index + 1}: ${(mockResults[index] * 100).toFixed(1)}%`
      ).join('\n')}`);
      
    } catch (error) {
      console.error('Error calculating equity:', error);
      alert('Error calculating equity. Please try again.');
    } finally {
      setIsCalculating(false);
    }
  };

  return (
    <div className="w-full h-screen bg-green-900 relative overflow-hidden">
      {/* Felt texture overlay */}
      <div className="absolute inset-0 opacity-20" style={{
        backgroundImage: `radial-gradient(circle at 20% 30%, rgba(255,255,255,0.1) 1px, transparent 1px),
                         radial-gradient(circle at 80% 70%, rgba(255,255,255,0.1) 1px, transparent 1px),
                         radial-gradient(circle at 40% 80%, rgba(255,255,255,0.1) 1px, transparent 1px)`,
        backgroundSize: '100px 100px, 150px 150px, 200px 200px'
      }}></div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex items-center gap-4 z-10">
        <div className="relative">
          <select
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
            className="appearance-none bg-gray-800 text-white px-4 py-2 pr-8 rounded-lg border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            {VARIANTS.map(v => (
              <option key={v.value} value={v.value}>{v.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 text-white w-4 h-4 pointer-events-none" />
        </div>
        
        <button
          onClick={calculateEquity}
          disabled={isCalculating}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <Calculator className="w-4 h-4" />
          {isCalculating ? 'Calculating...' : 'Calculate Equity'}
        </button>

        <button
          onClick={shuffleAndDeal}
          className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <Shuffle className="w-4 h-4" />
          New Hand
        </button>
      </div>

      {/* Main poker table */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-[800px] h-[500px]">
          {/* Table surface */}
          <div className="w-full h-full bg-green-800 rounded-[50%] border-8 border-yellow-900 shadow-2xl relative">
            {/* Inner felt area */}
            <div className="absolute inset-4 bg-green-700 rounded-[50%] shadow-inner"></div>
            
            {/* Player positions */}
            <PlayerPosition position="top" cards={playerHands.top} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 1" />
            <PlayerPosition position="topRight" cards={playerHands.topRight} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 2" />
            <PlayerPosition position="right" cards={playerHands.right} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 3" />
            <PlayerPosition position="bottomRight" cards={playerHands.bottomRight} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 4" />
            <PlayerPosition position="bottom" cards={playerHands.bottom} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 5" />
            <PlayerPosition position="bottomLeft" cards={playerHands.bottomLeft} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 6" />
            <PlayerPosition position="left" cards={playerHands.left} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 7" />
            <PlayerPosition position="topLeft" cards={playerHands.topLeft} onCardClick={handlePositionCardClick} selectedCards={selectedCards} label="Player 8" />
            
            {/* Board cards (center) */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="text-white text-sm mb-2 text-center">Board</div>
              <div className="flex gap-2">
                {board.map((card, index) => (
                  <Card
                    key={index}
                    card={card}
                    onClick={() => handleBoardCardClick(index)}
                    isSelected={false}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Available cards deck */}
      <div className="absolute top-4 left-4 max-w-[600px]">
        <div className="text-white text-sm mb-2">Available Cards (Click to select)</div>
        <div className="grid grid-cols-13 gap-1 bg-black bg-opacity-30 p-3 rounded-lg max-h-32 overflow-y-auto">
          {availableCards.map(card => (
            <Card
              key={card.id}
              card={card}
              onClick={() => handleCardSelection(card)}
              isSelected={selectedCards.includes(card.id)}
              size="small"
            />
          ))}
        </div>
        {selectedCards.length > 0 && (
          <button
            onClick={dealSelectedCards}
            className="mt-2 bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-sm"
          >
            Deal Selected Cards ({selectedCards.length})
          </button>
        )}
      </div>

      {/* Dead cards area */}
      <div className="absolute bottom-4 left-4 max-w-[600px]">
        <div className="text-white text-sm mb-2">Dead Cards</div>
        <div className="flex flex-wrap gap-1 bg-black bg-opacity-30 p-2 rounded-lg min-h-16">
          {deadCards.map((card, index) => (
            <Card
              key={index}
              card={card}
              isClickable={false}
              size="small"
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PokerTable;