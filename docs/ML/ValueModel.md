```mermaid
flowchart TD
  In[Poker Input 14x4x2]
  In --> HandGrid[Hand Grid 14x4x1]
  In --> BoardGrid[Board Grid 14x4x1]
  HandGrid --> ComboConcat[ComboConcatLayer: stack hand, board, hand_plus_board]
  BoardGrid --> ComboConcat[ComboConcatLayer: stack hand, board, hand_plus_board]
  ComboConcat --> CombinedGrid[Combined Grid 13x4x3]

  %% SIMPLE SUBGRAPH TITLE, NO PARENTHESES
  subgraph CardStateEncoder["CardStateEncoder"]
    HandEnc[Hand Encoder - CardSetEncoder]
    BoardEnc[Board Encoder - CardSetEncoder]
    CombinedEnc[Combined Encoder - CardSetEncoder]
  end

  HandGrid --> HandEnc --> HandEmb[Hand Embedding]
  BoardGrid --> BoardEnc --> BoardEmb[Board Embedding]
  CombinedGrid --> CombinedEnc --> ComboEmb[Combined Embedding]

  %% ANOTHER SIMPLE SUBGRAPH TITLE
  subgraph GridValueHeads["Grid Value Head - CardStateGridValueHead"]
    HandHead[Hand Value Head - GridValueHead]
    BoardHead[Board Value Head - GridValueHead]
    ComboHead[Combined Value Head - GridValueHead]
  end

  HandEmb --> HandHead --> HandValue[Hand Value]
  BoardEmb --> BoardHead --> BoardValue[Board Value]
  ComboEmb --> ComboHead --> CombinedValue[Combined Value]
