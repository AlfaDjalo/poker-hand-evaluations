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
