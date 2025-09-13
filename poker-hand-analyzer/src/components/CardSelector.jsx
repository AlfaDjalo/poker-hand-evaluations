import React, { useState } from "react";
import Card from "./Card";
import "../css/Card.css";
import "../css/CardSelector.css";

const CardSelector = ({ onSelectCard, excludedCards = [] }) => {
  // const [selectedCards, setSelectedCards] = useState(new Set());

  // Generate a standard 52-card deck grouped by suit
  const suits = ["S", "H", "D", "C"]; // Spades, Hearts, Diamonds, Clubs
  const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];

  const generateDeck = () => {
    return suits.map((suit) =>
      ranks.map((rank) => `${rank}${suit}`)
    );
  };

  const deck = generateDeck();

  const handleCardClick = (card) => {
    if (!excludedCards.includes(card)) {
      onSelectCard(card); // push selection back up
    }
  };

  // const handleCardClick = (card) => {
  //   const newSelected = new Set(selectedCards);
  //   if (newSelected.has(card)) {
  //     newSelected.delete(card);
  //   } else {
  //     newSelected.add(card);
  //   }
  //   setSelectedCards(newSelected);
  // };

  // Render one suit row
  const renderSuit = (suitCards, suitIndex) => (
    <div key={suitIndex} className="suitRow">
      {suitCards.map((card) => (
        <div key={card}>
          {excludedCards.includes(card) ? (
            <div className="poker-card-placeholder" />
          ) : (
            <Card
              card={card}
              hidden={false}
              scale={0.8}
              onClick={() => handleCardClick(card)}
              isClickable={true}
              isSelected={false}
            />
          )}
        </div>
      ))}
    </div>
  );
  // const renderSuit = (suitCards, suitIndex) => (
  //   <div key={suitIndex} className="suitRow">
  //     {suitCards.map((card) => (
  //       <div key={card}>
  //         {selectedCards.has(card) ? (
  //           <div className="poker-card-placeholder" />
  //         ) : (
  //           <Card
  //             card={card}
  //             hidden={false}
  //             scale={1.0}
  //             onClick={() => handleCardClick(card)}
  //             isClickable={true}
  //             isSelected={false}
  //           />
  //         )}
  //       </div>
  //     ))}
  //   </div>
  // );

  return (
    <div className="card-selector">
      <h2>Select Cards from Deck</h2>
      <div className="deckGrid">
        {deck.map((suitCards, suitIndex) => renderSuit(suitCards, suitIndex))}
      </div>
    </div>
  );
};

//   return (
//     <div className="card-selector">
//         <h2>Select Cards from Deck</h2>
//         <p>Selected: {selectedCards.scale} cards</p>

//         {/* Full deck grouped by suits */}
//         <div className="deckGrid">
//             {deck.map((suitCards, suitIndex) => renderSuit(suitCards, suitIndex))}
//         </div>

//         {/* Selected cards section */}
//         {selectedCards.scale > 0 && (
//             <div style={{ marginTop: "2rem" }}>
//             <h3>Selected Cards:</h3>
//             <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
//                 {Array.from(selectedCards).map((card) => (
//                 <Card
//                     card={card}
//                     hidden={false}
//                     scale={0.7}
//                     onClick={() => handleCardClick(card)}
//                     isClickable={true}
//                     isSelected={true}
//                 />
//                 ))}
//             </div>
//             </div>
//         )}
//     </div>
//   );
// };

export default CardSelector;
