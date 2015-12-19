function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

n = size(A2, 1);
A2 = [ones(n, 1) A2];
A3 = sigmoid(A2 * Theta2');

y1 = zeros(m, num_labels); 
for i = 1:m
  y1(i, y(i)) = 1;
end

%J = trace(-log(A3) * y1' - log(1 .- A3) * (1 .- y1')) / m;
J = trace(-log(A3') * y1 - log(1 .- A3') * (1 .- y1)) / m;

Theta1_r = Theta1(:, 2:end);
Theta2_r = Theta2(:, 2:end);

J = J + lambda*(trace(Theta1_r * Theta1_r') + trace(Theta2_r * Theta2_r'))/m/2;

delta3 = A3 - y1;
delta2 = (delta3 * Theta2) .* [ones(m, 1) sigmoidGradient(Z2)];
Delta2 = delta3' * A2;
Delta1 = delta2' * A1;
Delta1 = Delta1(2:end, :);

for i = 1:size(Delta2, 1)
	for j = 2:size(Delta2, 2)
		Delta2(i,j) = Delta2(i, j) + Theta2(i, j)*lambda;
	end
end

for i = 1:size(Delta1, 1)
	for j = 2:size(Delta1, 2)
		Delta1(i,j) = Delta1(i, j) + Theta1(i, j)*lambda;
	end
end

%Theta2_grad = Delta2 ./ m;
%Theta1_grad = Delta1 ./ m;

%k1 = size(Theta1_r, 1);
%k2 = size(X, 2);
%delta3 = zeros(m, num_labels);
%delta2 = zeros(m,k1+1); 

%delta_1 = zeros(m,);
%delta_2 = zeros(m,k1);

%for i = 1:m
%  a1 = X(i, :);
%  z2 = [1 a1] * Theta1'
%  a2 = sigmoid(z2);
%  z3 = [1 a2] * Theta2';
%  a3 = sigmoid(z3);
%  delta3(i, :) = a3 - y1(i, :); %???
%  delta2(i, :) = (delta3(i, :) * Theta2) .& sigmoidGradient([1 z2]);
%  delta2(i, :) = delta2(i, 2:end);
%  A2 = zeros(k1, num_labels);
%  for j = 1: k1
%	  A2(j, a2(j) = 1
%  end

%  delta_2(i, :) = delta_2(i, :) + delta3(i, :) * A2';
%  delta_1(i, :) = delta_1(i, :) + delta2(i, :) * A1';
%end

Theta2_grad = Delta2 ./ m;
Theta1_grad = Delta1 ./ m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
