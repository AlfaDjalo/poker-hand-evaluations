```mermaid
flowchart TD
  In[Poker Input 13x4x2]
  In --> HandGrid[Hand Grid 13x4x1]
  In --> BoardGrid[Board Grid 13x4x1]
  HandGrid --> ComboConcat[ComboConcatLayer: stack hand, board, hand_plus_board]
  ComboConcat --> CombinedGrid[Combined Grid 13x4x3]
  BoardGrid --> ComboConcat[ComboConcatLayer: stack hand, board, hand_plus_board]

  %% SIMPLE SUBGRAPH TITLE, NO PARENTHESES
  subgraph ComboModel["PokerComboModel - shared encoders"]
    HandEnc[Hand Encoder - PokerCNNEncoder]
    CombinedEnc[Combined Encoder - PokerCNNEncoder]
    BoardEnc[Board Encoder - PokerCNNEncoder]
  end

  HandGrid --> HandEnc --> HandEmb[Hand Embedding]
  CombinedGrid --> CombinedEnc --> ComboEmb[Combined Embedding]
  BoardGrid --> BoardEnc --> BoardEmb[Board Embedding]

  %% ANOTHER SIMPLE SUBGRAPH TITLE
  subgraph ValueHeads["PokerValueHeads"]
    HandHead[Hand Value Head]
    ComboHead[Combined Value Head]
    BoardHead[Board Value Head]
  end

  HandEmb --> HandHead --> HandValue[hand_value]
  ComboEmb --> ComboHead --> CombinedValue[combined_value]
  BoardEmb --> BoardHead --> BoardValue[board_value]
