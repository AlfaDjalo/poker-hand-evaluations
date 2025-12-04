import React, {useState, useEffect } from "react";

import CardSelector from './CardSelector';
import Hand from './Hand';
import EmbeddingChart from './EmbeddingChart';
import CombinedEmbeddingChart from './CombinedEmbeddingChart';
import "../css/ViewEmbeddings.css";
import { loadEmbeddings } from "../utils/api";
import { usePokerDnD } from "../hooks/usePokerDnD"
import { DndContext, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";

const ViewEmbeddings = () => {
    // States - match EquityCalculator format: { card: string, hidden: boolean }
    const [embeddingHands, setEmbeddingHands] = useState([
        {
            cards: [
                { card: "TS", hidden: false },
                { card: "9S", hidden: false },
                { card: "8S", hidden: false },
                { card: "7S", hidden: false },
                { card: "6S", hidden: false }
            ],
            embedding: null
        }
    ]);

    const [calculating, setCalculating] = useState(false);
    const [embeddingMode, setEmbeddingMode] = useState("5");
    const [viewMode, setViewMode] = useState("individual");   // "individual" | "combined"
    
    const sensors = useSensors(
        useSensor(PointerSensor, {
        activationConstraint: { distance: 100 },
        })
    );

    useEffect(() => {
        const targetSize = Number(embeddingMode);

        setEmbeddingHands(prev =>
            prev.map(handObj => {
                const currentCards = handObj.cards || [];

                if (currentCards.length > targetSize) {
                    return {
                        ...handObj,
                        cards: currentCards.slice(0, targetSize),
                        embedding: null
                    };
                } else if (currentCards.length < targetSize) {
                    return {
                        ...handObj,
                        cards: [...currentCards, ...Array(targetSize - currentCards.length).fill(null)],
                        embedding: null
                    };
                }

                return handObj;
            })
        );
    }, [embeddingMode]);

    const handleCardClick = (handIndex, cardIndex) => {
        console.log("card clicked", handIndex, cardIndex, embeddingHands[handIndex][cardIndex]);
        setEmbeddingHands(prev => {
            const copy = prev.map(h => ({
                ...h,
                cards: h.cards.map(c => c ? { ...c } : null)         
            }));
            copy[handIndex].cards[cardIndex] = null;
            return copy;
        });
    };

    const handleSlotClick = (handIndex, slotIndex) => {
        console.log("slot clicked", handIndex, slotIndex);
    //     setEmbeddingHands(prev => {
    //         const copy = prev.map(h => h.slice());
    //         copy[handIndex][slotIndex] = copy[handIndex][slotIndex] ? null : { card: "As", hidden: false };
    //         return copy;
    //     });
    }

    const handleAddHand = () => {
        const size = Number(embeddingMode);
        setEmbeddingHands(prev => [
            ...prev,
            {
                cards: Array(size).fill(null),
                embedding: null
            }
        ]);
    };

    const handleDuplicateHand = (index) => {
        // setEmbeddingHands(prev => [...prev, [...prev[index]]]);
        setEmbeddingHands(prev => [
            ...prev,
            {
                cards: prev[index].cards.map(card => card ? {...card} : null),
                embedding: null
            }
        ]);
    };

    const handleRemoveHand = (index) => {
        setEmbeddingHands(prev => prev.filter((_, i) => i !== index));
    }

    async function handleLoadEmbeddings() {
        setCalculating(true);

        // setEmbeddingHands(prev => prev.filter((_, i) => i !== index));

        // Clear embeddings immediately on new run
        setEmbeddingHands(prev =>
            prev.map((hands, index) => ({
                ...hands,
                embedding: null
            }))
        );
    
        try {
            const res = await loadEmbeddings({
                hands: embeddingHands.map(hand => hand.cards),
                mode: embeddingMode
            });
        //   setResult(res);
          
            setEmbeddingHands(prev =>
                prev.map((hand, index) => ({
                    ...hand,
                    embedding: res.embeddings[index]
                }))
            );

        } catch (err) {
          console.error("Load failed:", err);
        } finally {
          setCalculating(false);
        }
    }
    
    const handleCardSelected = (card) => {
        // Implement card selection logic if needed
    }

    const { handleDragStart, handleDragEnd } = usePokerDnD({
        hands: embeddingHands,
        setHands: setEmbeddingHands,
        boardCards: null,  // No board in this view
        setBoardCards: null
    });

    const usedCards = [];

    // Render
    return (
        <DndContext
            sensors={sensors}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            // onDragStart={event => {
            //     console.log("Drag event started.", event);
            // }}
            // onDragEnd={event => {
            //     console.log("Drag event ended.", event);
            // }}
        >
            <div className="view-embeddings">
                <h2>View Embeddings</h2>

                <CardSelector
                    onSelectCard={handleCardSelected}
                    usedCards = {usedCards}
                />

                <div className="mb-4">
                    <select
                        value={embeddingMode}
                        onChange={e => setEmbeddingMode(e.target.value)}
                        className="ve-select mr-4"
                    >
                        <option value="2">Hole Cards (2)</option>
                        <option value="3">Board (3)</option>
                        <option value="5">Combined (5)</option>
                    </select>
                </div>

                <div className="mb-4">
                    <button
                        onClick={handleAddHand}
                        className="ve-button mr-2"
                    >
                        + Add hand
                    </button>
                    <span className="text-sm text-gray-600">Dummy hands shown below</span>
                </div>

                <div className="mb-4">
                    <button
                        onClick={handleLoadEmbeddings}
                        className="ve-button mr-2"
                    >
                        Load Embeddings
                    </button>
                    <span className="text-sm text-gray-600">Dummy hands shown below</span>
                </div>

                <div className="mb-4">
                    <button
                        onClick={() => setViewMode("individual")}
                        className={`ve-button mr-2 ${viewMode === "individual" ? "active" : ""}`}
                    >
                        Individual charts
                    </button>

                    <button
                        onClick={() => setViewMode("combined")}
                        className={`ve-button ${viewMode === "combined" ? "active" : ""}`}
                    >
                        Combined chart
                    </button>

                    <span className="text-sm text-gray-600 ml-3">
                        Switch how embeddings are visualised
                    </span>
                </div>

                <div className="embedding-hands-container">
                    {embeddingHands.map((handObj, i) => (
                        <div key={i} className="embedding-hand-block flex flex-col items-center">
                            <div className="mb-2">
                                <Hand 
                                    seatNumber={i}
                                    cards={handObj.cards}
                                    scale={1}
                                    showSlots={true}
                                    activeSlot={null}
                                    onCardClick={(cardIndex) => handleCardClick(i, cardIndex)}
                                    onSlotClick={(slotIndex) => handleSlotClick(i, slotIndex)}
                                    maxCards={Number(embeddingMode)}
                                    />
                            </div>
                            
                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleDuplicateHand(i)}
                                    className="ve-button-secondary"
                                >
                                    Duplicate
                                </button>
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleRemoveHand(i)}
                                    className="ve-button-secondary"
                                >
                                    Remove
                                </button>
                            </div>

                            {viewMode === "individual"
                            ? <EmbeddingChart embedding={handObj.embedding} />
                            : null}


                            {/* <EmbeddingChart embedding={handObj.embedding} /> */}
                        </div>
                    ))}
                </div>
            </div>
            {viewMode === "combined" && (
                <div className="combined-chart-wrapper mt-8 w-full">
                    <h3 className="text-lg font-semibold mb-2">Combined Embeddings</h3>
                    <CombinedEmbeddingChart hands={embeddingHands} />
                </div>
            )}
        </DndContext>
    );
};

export default ViewEmbeddings;
