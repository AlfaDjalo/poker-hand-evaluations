```mermaid
flowchart TD

%% =======================
%% TOP-LEVEL INPUTS
%% =======================
subgraph Inputs["Pairwise Inputs"]
  X1[(x1)]
  X2[(x2)]
end

%% X1 splits
X1 --> H1[Hand 1 Grid 13x4x1]
X1 --> B1[Board 1 Grid 13x4x1]
X1 --> C1[Combo 1 Grid hand1 + board1]

%% X2 splits
X2 --> H2[Hand 2 Grid 13x4x1]
X2 --> B2[Board 2 Grid 13x4x1]
X2 --> C2[Combo 2 Grid hand2 + board2]


%% =========================================
%% SHARED ENCODERS (PokerComboModel)
%% =========================================
subgraph Encoders["PokerComboModel (shared encoders)"]
  HE[Hand Encoder: PokerCNNEncoder]
  BE[Board Encoder: PokerCNNEncoder]
  CE[Combined Encoder: PokerCNNEncoder]
end

H1 --> HE --> H1E[Hand 1 Emb]
B1 --> BE --> B1E[Board 1 Emb]
C1 --> CE --> C1E[Combo 1 Emb]

H2 --> HE --> H2E[Hand 2 Emb]
B2 --> BE --> B2E[Board 2 Emb]
C2 --> CE --> C2E[Combo 2 Emb]


%% =========================================
%% VALUE HEADS (PokerValueHeads)
%% =========================================
subgraph ValueHeads["PokerValueHeads (shared heads)"]
  HH[Hand Value Head]
  BH[Board Value Head]
  CH[Combined Value Head]
end

H1E --> HH --> HV1[Hand Value 1]
B1E --> BH --> BV1[Board Value 1]
C1E --> CH --> CV1[Combined Value 1]

H2E --> HH --> HV2[Hand Value 2]
B2E --> BH --> BV2[Board Value 2]
C2E --> CH --> CV2[Combined Value 2]


%% =========================================
%% PAIRWISE SUBTRACTION
%% =========================================
subgraph PairwiseDiff["PairwiseModel Diff Layers"]
  HDiff[Subtract: hand_v1 - hand_v2]
  BDiff[Subtract: board_v1 - board_v2]
  CDiff[Subtract: combined_v1 - combined_v2]
end

HV1 --> HDiff
HV2 --> HDiff

BV1 --> BDiff
BV2 --> BDiff

CV1 --> CDiff
CV2 --> CDiff


%% =========================================
%% FINAL OUTPUTS
%% =========================================
HDiff --> HProb["Hand Comparison Prob (sigmoid)"]
BDiff --> BProb["Board Comparison Prob (sigmoid)"]
CDiff --> CProb["Combined Comparison Prob (sigmoid)"]
