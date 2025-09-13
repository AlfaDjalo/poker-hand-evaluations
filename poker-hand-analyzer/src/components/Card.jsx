import React from "react";
import "../css/Card.css";

const cardBaseURL = "https://deckofcardsapi.com/static/img";
const backCardURL = "https://deckofcardsapi.com/static/img/back.png";

const Card = ({ 
  card = null, 
  hidden = false, 
  scale = 1.0, 
  isClickable = true, 
  isSelected = false,
  onClick 
}) => {
  const normalizeForImage = (card) => {
    if (!card || card.length !== 2) return null;
    const rank = card[0] === "T" ? "0" : card[0];
    const suit = card[1].toUpperCase();
    return `${rank}${suit}`;
  };

  // Clamp scale to reasonable values
  const scaleFactor = Math.max(0.1, Math.min(3.0, Number(scale) || 1.0));

  if (!card) {
    // Render empty slot
    return (
      <span 
        className="cardEmpty" 
        style={{ transform: `scale(${scaleFactor})` }}
      />
    );
  }

  // If card is hidden, show card back
  if (hidden) {
    return (
      <img
        src={backCardURL}
        alt="Hidden card"
        className={`card ${isSelected ? 'selected' : ''}`}
        onClick={isClickable ? onClick : undefined}
        style={{ 
          cursor: isClickable ? 'pointer' : 'default',
          transform: `scale(${scaleFactor})`
        }}
      />
    );
  }

  const apiCard = normalizeForImage(card);
  
  // Build class names properly
  const classNames = `card ${isSelected ? 'selected' : ''}`;

  return (
    <img
      src={`${cardBaseURL}/${apiCard}.png`}
      alt={card}
      className={classNames}
      onClick={isClickable ? onClick : undefined}
      style={{ 
        cursor: isClickable ? 'pointer' : 'default',
        transform: `scale(${scaleFactor})`
      }}
      onError={(e) => {
        console.error(`Failed to load card image for ${card}`);
        // Create a fallback div instead of modifying the image
        const fallback = document.createElement('div');
        fallback.className = 'cardEmpty';
        fallback.textContent = card;
        fallback.style.cssText = `
          display: flex; 
          align-items: center; 
          justify-content: center; 
          background: #f0f0f0; 
          border: 1px solid #ccc;
          border-radius: 0.25rem;
          font-size: 12px;
          font-weight: bold;
          cursor: ${isClickable ? 'pointer' : 'default'};
          transform: scale(${scaleFactor});
        `;
        if (isClickable && onClick) {
          fallback.addEventListener('click', onClick);
        }
        e.target.parentNode.replaceChild(fallback, e.target);
      }}
    />
  );
};

export default Card;