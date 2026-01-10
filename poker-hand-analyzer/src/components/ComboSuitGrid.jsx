import React, { useMemo } from "react";

const SUITS = ["s", "h", "d", "c"];
const SUIT_LABELS = { s: "♠", h: "♥", d: "♦", c: "♣" };

export default function ComboSuitGrid({ combos, mode }) {
  /**
   * Build a 4x4 grid like the hand grid:
   * grid[rowSuit][colSuit] = avg probability or null
   */
  const grid = useMemo(() => {
    const table = {};
    SUITS.forEach(r => {
      table[r] = {};
      SUITS.forEach(c => {
        table[r][c] = [];
      });
    });

    combos.forEach(({ cards, probability, value }) => {
        const s1 = cards[0].slice(-1);
        const s2 = cards[1].slice(-1);
        const dataPoint = mode === 'probs' ? probability : value;
        table[s1][s2].push(dataPoint);
    });

    // combos.forEach(({ cards, probability }) => {
    //   const s1 = cards[0].slice(-1);
    //   const s2 = cards[1].slice(-1);
    //   table[s1][s2].push(probability);
    // });

    // average
    SUITS.forEach(r => {
      SUITS.forEach(c => {
        const arr = table[r][c];
        table[r][c] = arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
      });
    });

    return table;
  }, [combos]);

  const getCellStyle = (value, rowSuit, colSuit) => {
    const invalid = value == null;

    return {
      backgroundColor: invalid
        ? "#f3f4f6"
        : `rgba(${value > 0.5 ? "34,197,94" : "239,68,68"}, ${0.15 + 0.5 * value})`,
      color: invalid ? "#9ca3af" : "#111827",
      border: "1px solid #e5e7eb",
      cursor: invalid ? "default" : "pointer",
    };
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-3">Suit Grid</h3>

        <div className="suit-grid">
        {SUITS.map((rowSuit) =>
            SUITS.map((colSuit) => {
            const value = grid[rowSuit][colSuit];

            return (
                <div
                key={`${rowSuit}-${colSuit}`}
                className="suit-cell"
                style={{
                    backgroundColor:
                    value == null
                        ? "#374151"
                        : mode === 'probs'
                        ? value >= 0.75
                            ? "#16a34a"
                            : value >= 0.5
                            ? "#22c55e"
                            : value >= 0.25
                            ? "#eab308"
                            : "#dc2626"
                        : value > 0
                        ? "#16a34a"
                        : value < 0
                        ? "#dc2626"
                        : "#eab308",
                    color: value == null ? "#9ca3af" : "white",
                }}
                // style={{
                //     backgroundColor:
                //     value == null
                //         ? "#374151"
                //         : value >= 0.5
                //         ? `rgba(34,197,94,${0.3 + value * 0.5})`
                //         : `rgba(239,68,68,${0.3 + (1 - value) * 0.5})`,
                //     color: value == null ? "#9ca3af" : "white",
                // }}
                >
                <div className="text-sm font-bold">
                    {SUIT_LABELS[rowSuit]}
                    {SUIT_LABELS[colSuit]}
                </div>
                <div className="text-xs mt-1">
                    {value == null ? "—" : mode === 'probs' ? `${Math.round(value * 100)}%` : value.toFixed(2)}
                </div>                
                {/* <div className="text-xs mt-1">
                    {value == null ? "—" : `${Math.round(value * 100)}%`}
                </div> */}
                </div>
            );
            })
        )}
        </div>

      {/* <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "4px",
        }}
      >
        {SUITS.map((rowSuit) =>
          SUITS.map((colSuit) => {
            const value = grid[rowSuit][colSuit];

            return (
              <div
                key={`${rowSuit}-${colSuit}`}
                className="flex flex-col items-center justify-center rounded-md"
                style={{
                  height: 64,
                  ...getCellStyle(value, rowSuit, colSuit),
                }}
              >
                <div className="font-bold text-lg">
                  {SUIT_LABELS[rowSuit]}
                  {SUIT_LABELS[colSuit]}
                </div>

                <div className="text-sm mt-1">
                  {value == null ? "—" : `${Math.round(value * 100)}%`}
                </div>
              </div>
            );
          })
        )}
      </div> */}

      <p className="mt-2 text-xs text-gray-400 text-center">
        Rows = first card suit · Columns = second card suit
      </p>
    </div>
  );
}
