import React from "react";
import Card from "./Card";
import "../css/Hand.css";

const Hand = ({ cards = [], scale = 0.7, overlap = true, overlapOffset = 20 }) => {
  if (!cards.length) {
    return <div className="hand empty">No cards</div>;
  }

  return (
    <div className="hand">
      {cards.map((cardData, index) => {
        const { card, hidden } = cardData;
        
        const cardStyle = overlap && index > 0 ? {
          marginLeft: `-${overlapOffset}px`,
          zIndex: cards.length - index  // Later cards appear on top
        } : {
          zIndex: cards.length - index
        };

        return (
          <div 
            key={index} 
            className="hand-card" 
            style={cardStyle}
          >
            <Card
              card={card}
              hidden={hidden}
              scale={scale}
              isClickable={false}  // Hand cards typically not clickable
            />
          </div>
        );
      })}
    </div>
  );
};

// const adjustedOffset = scale;
  // const adjustedOffset = getOverlapOffset(size);

//   return (
//     <div className="hand">
//       {cards.map((c, idx) => (
//         <div
//           key={idx}
//           className="hand-card"
//           style={{
//             marginLeft: overlap && idx > 0 ? `-${adjustedOffset}px` : 0,
//           }}
//         >
//           <Card 
//             card={c.card} 
//             hidden={c.hidden}
//             scale={scale}
//             // onClick={() => handleCardClick(card)}          
//             isClickable={false}
//             isSelected={true}
//           />          
//         </div>
//       ))}
//     </div>
//   );
// };

export default Hand;