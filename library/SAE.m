function [ W ] = SAE(X, S, lambda)
    % SAE is Semantic Auto-encoder
    % Inputs:
    %    X: dxN data matrix.
    %    S: kxN semantic matrix.
    %    lambda: regularisation parameter.
    %
    % Return: 
    %    W: kxd projection matrix.

    A = S*S';
    B = lambda * X*X';
    C = (1 + lambda) * S*X'; 
    W = sylvester(A,B,C);
end

