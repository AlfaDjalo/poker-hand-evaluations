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

export async function loadEmbeddings({ hands, mode }) {
    const normalizedHands = hands.map(h => h.map(normalizeCardValue));
    
    console.log("Hands:", normalizedHands);
    
    const response = await fetch(`${API_URL}/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            embeddingHands: normalizedHands,
            mode: mode
        })
    });

    if (!response.ok) throw new Error("Embedding evaluation failed")

    const data = await response.json();
    return data;
}

export async function fetchPushFoldGrid({ stackBB, position, mode }) {
    console.log("Fetching push-fold grid:", { stackBB, position, mode });
    
    const response = await fetch(
        `${API_URL}/pushfold/grid?stack_bb=${stackBB}&position=${position}&mode=${mode}`,
        {
            method: "GET",
            headers: { "Content-Type": "application/json" }
        }
    );

    if (!response.ok) {
        const errorText = await response.text();
        console.error("Push-fold grid error:", errorText);
        throw new Error(`Failed to fetch push-fold grid: ${response.status}`);
    }

    const data = await response.json();
    console.log("Push-fold grid data received:", data);
    return data;
}

function normalizeCardValue(c) {
  return typeof c === "string" ? c : c?.card;
}