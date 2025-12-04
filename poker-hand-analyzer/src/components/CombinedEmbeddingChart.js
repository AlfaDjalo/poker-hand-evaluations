import React, { useMemo } from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

const COLORS = [
    "#8884d8",
    "#82ca9d",
    "#ff7300",
    "#413ea0",
    "#ff0000",
    "#00bcd4",
    "#8bc34a",
    "#ffc107",
];

const CombinedEmbeddingChart = ({ hands }) => {
    // Always call hooks — compute raw memo
    const prepared = useMemo(() => {
        return hands.map((h, i) => ({
            index: i,
            embedding: h.embedding,
        }));
    }, [hands]);

    // Filter *after* memo ensuring no hook-conditional errors
    const validHands = prepared.filter(h => Array.isArray(h.embedding));

    // If none have embeddings yet: render placeholder div
    if (validHands.length === 0) {
        return <div>No embeddings to display.</div>;
    }

    // Build Recharts-compatible dataset
    const MAX_DIM = Math.max(...validHands.map(h => h.embedding.length));

    const data = Array.from({ length: MAX_DIM }, (_, dim) => {
        const row = { dim: dim.toString() };

        validHands.forEach((h, i) => {
            const value = h.embedding[dim];
            row[`hand_${i}`] = typeof value === "number" ? value : null;
        });

        return row;
    });

    return (
        <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
                <BarChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <XAxis
                        dataKey="dim"
                        type="category"
                        label={{ value: "Embedding Dimension", position: "insideBottom", dy: 10 }}
                    />

                    <YAxis
                        type="number"
                        domain={[-1, 1]}
                        tickFormatter={(v) => v.toFixed(1)}
                        label={{
                            value: "Value",
                            angle: -90,
                            position: "insideLeft",
                            dx: -5,
                        }}
                    />

                    <Tooltip />
                    <Legend />

                    {validHands.map((_, i) => (
                        <Bar
                            key={i}
                            dataKey={`hand_${i}`}
                            fill={COLORS[i % COLORS.length]}
                            barSize={8}
                        />
                    ))}
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default CombinedEmbeddingChart;
