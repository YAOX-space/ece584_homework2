import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # TODO: Return the converted HardTan
        l = BoundHardTanh()
        return l
        
    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        # TODO: Implement the linear lower and upper bounds for HardTanH you derived in Problem 4.2.
        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes(coefficients)/intercepts(bias)
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """
        # Initialize 
        upper_slope = torch.zeros_like(preact_lb)
        upper_intercept = torch.zeros_like(preact_lb)
        
        lower_slope = torch.zeros_like(preact_lb)
        lower_intercept = torch.zeros_like(preact_lb)
        
        uA = None
        #  preact_lb = None
        ubias = 0
        lbias = 0
            
        mask1 = (preact_lb >= 1)
        upper_slope[mask1] = 0
        upper_intercept[mask1] = 1
        lower_slope[mask1] = 0
        lower_intercept[mask1] = 1
            
        mask2 = (preact_ub <= -1)
        upper_slope[mask2] = 0
        upper_intercept[mask2] = -1
        lower_slope[mask2] = 0
        lower_intercept[mask2] = -1
            
        mask3 = ((preact_lb <= -1) & (preact_ub >= 1))
        upper_slope[mask3] = 1
        upper_intercept[mask3] = 0
        lower_slope[mask3] = 1
        lower_intercept[mask3] = 0
            
        mask4 = ((preact_lb >= -1) & (preact_ub <= 1))
        upper_slope[mask4] = 1
        upper_intercept[mask4] = 0
        lower_slope[mask4] = 1
        lower_intercept[mask4] = 0
          
        mask5 = ((preact_lb <= -1) & (preact_ub >= -1) & (preact_ub <= 1))
            
        denom = preact_ub - preact_lb
        # in case of denom == 0
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            
        upper_slope[mask5] = (preact_ub + 1.0) / denom
        upper_intercept[mask5] = -1.0 - upper_slope * preact_lb
        lower_slope[mask5] = torch.zeros_like(preact_lb)
        lower_intercept[mask5] = -torch.ones_like(preact_lb)
            
        mask6 = ((preact_lb >= -1) & (preact_lb <= 1) & (preact_ub >= 1))
        denom = preact_ub - preact_lb
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            
        upper_slope[mask6] = torch.zeros_like(preact_lb)
        upper_intercept[mask6] = torch.ones_like(preact_lb)
        lower_slope[mask6] = (1.0 - preact_lb) / denom
        lower_intercept[mask6] = preact_lb - lower_slope * preact_lb
            
        
        # Apply the bounds to the coefficient matrices
        if last_uA is not None:
            # For upper bound: uA = upper_slope * last_uA
            uA = upper_slope.unsqueeze(1) * last_uA
            # Bias term: ubias = upper_intercept * sum(last_uA)
            mult_uA = last_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_intercept.view(upper_intercept.size(0), -1, 1)).squeeze(-1)
            
        if last_lA is not None:
            # For lower bound: lA = lower_slope * last_lA  
            lA = lower_slope.unsqueeze(1) * last_lA
            # Bias term: lbias = lower_intercept * sum(last_lA)
            mult_lA = last_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(lower_intercept.view(lower_intercept.size(0), -1, 1)).squeeze(-1)

        # You should return the linear lower and upper bounds after propagating through this layer.
        # Upper bound: uA is the coefficients, ubias is the bias.
        # Lower bound: lA is the coefficients, lbias is the bias.

        return uA, ubias, lA, lbias

