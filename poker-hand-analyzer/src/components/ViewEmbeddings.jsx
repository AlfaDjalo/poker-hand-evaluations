import React, {useState } from "react";

import CardSelector from './CardSelector';
import Hand from './Hand';
import "../css/ViewEmbeddings.css";
import { usePokerDnD } from "../hooks/usePokerDnD"

import { DndContext, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";

const ViewEmbeddings = () => {
    // States - match EquityCalculator format: { card: string, hidden: boolean }
    const [embeddingHands, setEmbeddingHands] = useState([
        [
            { card: "AS", hidden: false },
            { card: "JD", hidden: false },
            { card: "8D", hidden: false },
            { card: "6C", hidden: false },
            { card: "4S", hidden: false }
        ],
        [
            { card: "KH", hidden: false },
            { card: "QS", hidden: false },
            { card: "TD", hidden: false },
            { card: "9C", hidden: false },
            null
        ],
        [null, null, null, null, null],
    ]);

    const sensors = useSensors(
        useSensor(PointerSensor, {
        activationConstraint: { distance: 100 },
        })
    );

    const handleCardClick = (handIndex, cardIndex) => {
        console.log("card clicked", handIndex, cardIndex, embeddingHands[handIndex][cardIndex]);
        setEmbeddingHands(prev => {
            const copy = prev.map(h => h.slice());
            copy[handIndex][cardIndex] = null;
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
        setEmbeddingHands(prev => [...prev, [null, null, null, null, null]]);
    };

    const handleDuplicateHand = (index) => {
        setEmbeddingHands(prev => [...prev, [...prev[index]]]);
    };

    const handleRemoveHand = (index) => {
        setEmbeddingHands(prev => prev.filter((_, i) => i !== index));
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
                        <button
                            onClick={handleAddHand}
                            className="ve-button mr-2"
                        >
                            + Add hand
                        </button>
                        <span className="text-sm text-gray-600">Dummy hands shown below</span>
                    </div>

                    <div className="embedding-hands-container">
                        {embeddingHands.map((hand, i) => (
                            <div key={i} className="embedding-hand-block flex flex-col items-center">
                                <div className="mb-2">
                                    <Hand 
                                        seatNumber={i}
                                        cards={hand}
                                        scale={1}
                                        showSlots={true}
                                        activeSlot={null}
                                        onCardClick={(cardIndex) => handleCardClick(i, cardIndex)}
                                        onSlotClick={(slotIndex) => handleSlotClick(i, slotIndex)}
                                        maxCards={5}
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
                            </div>
                        ))}
                    </div>
                </div>
        </DndContext>
    );
};

export default ViewEmbeddings;
