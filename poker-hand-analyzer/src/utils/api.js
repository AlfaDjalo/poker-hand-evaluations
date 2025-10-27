const API_URL = "http://127.0.0.1:8000"; // change later if needed

export async function evaluateHand(playerHands, board) {
  console.log("PlayerHands:", playerHands);
  console.log("Board:", board);

  const response = await fetch(`${API_URL}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ playerHands, board }),
  });

  if (!response.ok) {
    throw new Error("Failed to evaluate");
  }

  return await response.json();
}

export async function evaluateShowdown(playerHands, board) {
    const normalizedHands = playerHands.map(hand => hand.map(normalizeCardValue));
    const normalizedBoard = board.map(normalizeCardValue);
    
    console.log("PlayerHands:", normalizedHands);
    console.log("Board:", normalizedBoard);
    
    const response = await fetch(`${API_URL}/showdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            playerHands: normalizedHands,
            board: normalizedBoard
        })
    //    body: JSON.stringify({ normalizedHands, normalizedBoard })
    });

    if (!response.ok) throw new Error("Showdown evaluation failed")

    const data = await response.json();
    return data;
}

function normalizeCardValue(c) {
  return typeof c === "string" ? c : c?.card;
}