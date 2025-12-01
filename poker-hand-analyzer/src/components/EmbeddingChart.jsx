import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const EmbeddingChart = ({ embedding }) => {
  if (!embedding || embedding.length === 0) {
    return (
      <div style={{ fontSize: "0.85rem", color: "#666", marginTop: "4px" }}>
        No embedding loaded
      </div>
    );
  }

  // Convert embedding vector → chart-friendly format
  const data = embedding.map((value, index) => ({
    dim: index,
    value,
  }));

  return (
    <div style={{ width: "100%", height: Math.max(120, embedding.length * 12) }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, bottom: 5, left: 5, right: 5 }}
        >
          <YAxis
            type="category"
            dataKey="dim"
            width={30}
            tick={{ fontSize: 9 }}
          />
          <XAxis
            type="number"
            tick={{ fontSize: 10 }}
            domain={["auto", "auto"]}
          />
          <Tooltip />
          <Bar
            dataKey="value"
            fill="#82ca9d"
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmbeddingChart;
