/**
 * Creates a shuffled 52-card deck
 * @returns {Array<string>} Shuffled array of card codes
 */
export const generateDeck = () => {
  const suits = ["S", "H", "D", "C"];
  const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
  const deck = suits.flatMap(suit => ranks.map(rank => `${rank}${suit}`));
  return shuffleArray([...deck]);
};

/**
 * Shuffles an array using Fisher-Yates algorithm
 * @param {Array} array - Array to shuffle
 * @returns {Array} Shuffled array
 */
export const shuffleArray = (array) => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

/**
 * Validates a card code format
 * @param {string} card - Card code to validate
 * @returns {boolean} Whether the card code is valid
 */
export const isValidCard = (card) => {
  if (!card || typeof card !== 'string' || card.length !== 2) return false;
  
  const rank = card[0];
  const suit = card[1];

  const suits = ["S", "H", "D", "C"];
  const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
  
  return validRanks.includes(rank) && validSuits.includes(suit);
};