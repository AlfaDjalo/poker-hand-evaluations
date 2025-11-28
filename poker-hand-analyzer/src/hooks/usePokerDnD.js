import { useState } from "react";

/**
 * Custom hook for poker drag-and-drop functionality
 * Handles card movement between hands, board, and trash
 * 
 * @param {Object} config
 * @param {Object} config.players - Player state object (for EquityCalculator)
 * @param {Function} config.setPlayers - Player state setter
 * @param {Array} config.hands - Hands array (for ViewEmbeddings)
 * @param {Function} config.setHands - Hands state setter
 * @param {Array} config.boardCards - Board cards array
 * @param {Function} config.setBoardCards - Board cards setter
 * @param {Function} [config.onCardMoved] - Optional callback after card movement
 */

export function usePokerDnD({
    players,
    setPlayers,
    hands,
    setHands,
    boardCards,
    setBoardCards,
    onCardMoved
}) {
    const handleDragStart = (event) => {
        console.log("Drag started:", event);
    };

    const handleDragEnd = (event) => {
        const { active, over } = event;

        if (!over) {
            console.log("Dropped outside any droppable");
            return;
        }

        const draggedCard = active.data.current?.card;
        const sourceData = active.data.current;
        const targetData = over.data.current;

        if (!draggedCard || !targetData) {
            console.log("No dragged card or target data");
            return;
        }   
        
        console.log("🃏 Dropped", draggedCard, "Source:", sourceData, "Target:", targetData);
        
        if (onCardMoved) {
            onCardMoved({ draggedCard, sourceData, targetData });
        }

        // Handle selector as source (adding new cards)
        if (sourceData.variant === 'selector') {
            handleSelectorDrop(draggedCard, targetData);
            return;
        }

        // Handle different move types
        if (sourceData.variant === targetData.variant) {
            handleSameVariantSwap(sourceData, targetData);
        } else if (
            (sourceData.variant === "board" && targetData.variant === "player") ||
            (sourceData.variant === "player" && targetData.variant === "board")
        ) {
            handleBoardHandSwap(sourceData, targetData);
        } else if (targetData.variant === "selector") {
            handleTrash(sourceData);
        }
    };

    // Handle dropping from selector (placing new cards)
    const handleSelectorDrop = (card, targetData) => {
        if (targetData.variant === "board" && setBoardCards) {
            setBoardCards(prev => {
                const newBoard = [...prev];
                newBoard[targetData.index] = { card, hidden: false };
                return newBoard;
            });
        } 
        else if (targetData.variant === "player") {
            const seat = targetData.location;
            const slotIndex = targetData.index;

            // if ((players && setPlayers) || (hands && setHands)) {
            if (players && setPlayers) {
                
                // EquityCalculator format
                setPlayers(prev => {
                    const updated = { ...prev };
                    const hand = [...updated[seat].hand];
                    hand[slotIndex] = { card, hidden: false };
                    updated[seat] = { ...updated[seat], hand };
                    return updated;
                });
            } else if (hands && setHands) {
                // ViewEmbeddings format
                setHands(prev => {
                    // Before applying the update:
                    const targetHand = prev[seat];
    
                    // Determine the card we want to drop:
                    const { rank, suit } = card;
    
                    // Check for duplicates in the same hand (except the slot we’re replacing)
                    const alreadyUsed = targetHand.some((slot, i) =>
                        slot?.card === card
                        // i !== slotIndex &&
                        // slot?.card?.rank === rank &&
                        // slot?.card?.suit === suit
                    );
    
                    console.log("targetHand: ", targetHand);
                    console.log("Card: ", card);

                    if (alreadyUsed) {
                        // Reject update (just return prev unchanged)
                        return prev;
                    }

                    const newHands = prev.map(h => [...h]);
                    newHands[seat][slotIndex] = { card, hidden: false };
                    return newHands;
                });
            }
        }
    };

    // Swap within same component (board-to-board or hand-to-hand)
    const handleSameVariantSwap = (sourceData, targetData) => {
        if (sourceData.variant === "board" && setBoardCards) {
            setBoardCards(prev => {
                const newBoard = [...prev];
                [newBoard[sourceData.index], newBoard[targetData.index]] =
                [newBoard[targetData.index], newBoard[sourceData.index]];
                return newBoard;
            });
        }
        else if (sourceData.variant === "player") {
            const sourceSeat = sourceData.location;
            const targetSeat = targetData.location;

            // For EquityCalculator (players object)
            if (players && setPlayers) {
                setPlayers(prev => {
                    const updated = { ...prev };

                    if (sourceSeat === targetSeat) {
                        // Same hand
                        const hand = [...updated[sourceSeat].hand];
                        [hand[sourceData.index], hand[targetData.index]] =
                        [hand[targetData.index], hand[sourceData.index]];
                        updated[sourceSeat] = { ...updated[sourceSeat], hand };
                    } else {
                        // Different hands
                        const sourceHand = [...updated[sourceSeat].hand];
                        const targetHand = [...updated[targetSeat].hand];
                        [sourceHand[sourceData.index], targetHand[targetData.index]] =
                        [targetHand[targetData.index], sourceHand[sourceData.index]];
                        updated[sourceSeat] = { ...updated[sourceSeat], hand: sourceHand };
                        updated[targetSeat] = { ...updated[targetSeat], hand: targetHand };
                    }

                    return updated;
                });
            }

            // For ViewEmbeddings (hands array)
            else if (hands && setHands) {
                setHands(prev => {
                    const newHands = prev.map(h => [...h]);

                    if (sourceSeat === targetSeat) {
                        [newHands[sourceSeat][sourceData.index], newHands[targetSeat][targetData.index]] =
                        [newHands[targetSeat][targetData.index], newHands[sourceSeat][sourceData.index]]
                    } else {
                        [newHands[sourceSeat][sourceData.index], newHands[targetSeat][targetData.index]] =
                        [newHands[targetSeat][targetData.index], newHands[sourceSeat][sourceData.index]]
                    }

                    return newHands;
                });
            }
        }
    };

    // Swap between board and hand
    const handleBoardHandSwap = (sourceData, targetData) => {
        const boardIndex = sourceData.variant === "board" ? sourceData.index : targetData.index;
        const handSeat = sourceData.variant === "player" ? sourceData.location : targetData.location;
        const handIndex = sourceData.variant === "player" ? sourceData.index : targetData.index;

        if (players && setPlayers) {
        // EquityCalculator format
        const boardCard = boardCards[boardIndex];
        const handCard = players[handSeat].hand[handIndex];

        setBoardCards(prev => {
            const newBoard = [...prev];
            newBoard[boardIndex] = handCard;
            return newBoard;
        });

        setPlayers(prev => {
            const updated = { ...prev };
            const hand = [...updated[handSeat].hand];
            hand[handIndex] = boardCard;
            updated[handSeat] = { ...updated[handSeat], hand };
            return updated;
        });
        } else if (hands && setHands) {
        // ViewEmbeddings format
        const boardCard = boardCards[boardIndex];
        const handCard = hands[handSeat][handIndex];

        setBoardCards(prev => {
            const newBoard = [...prev];
            newBoard[boardIndex] = handCard;
            return newBoard;
        });

        setHands(prev => {
            const newHands = prev.map(h => [...h]);
            newHands[handSeat][handIndex] = boardCard;
            return newHands;
        });
        }
    };

    // Remove card (drop on selector)
    const handleTrash = (sourceData) => {
        console.log("🗑️ Dropped on selector (trash)");

        if (sourceData.variant === "board" && setBoardCards) {
        setBoardCards(prev => {
            const newBoard = [...prev];
            newBoard[sourceData.index] = null;
            return newBoard;
        });
        } 
        else if (sourceData.variant === "player") {
        const seat = sourceData.location;

        if (players && setPlayers) {
            setPlayers(prev => {
                const updated = { ...prev };
                const hand = [...updated[seat].hand];
                hand[sourceData.index] = null;
                updated[seat] = { ...updated[seat], hand };
                return updated;
            });
        } else if (hands && setHands) {
            setHands(prev => {
                const newHands = prev.map(h => [...h]);
                newHands[seat][sourceData.index] = null;
                return newHands;
            });
        }
        }
    };

    return { handleDragStart, handleDragEnd };
}

function parseLocation(raw: string) {
    const [zone, index] = raw.split("-");

    return {
        zone,
        index: index ? Number(index) : null,
    };
}
