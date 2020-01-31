import numpy as np
from mvpa_itab.script.meg.jakko_converter import n_subjects


def prevalence(ds, n_p2=1e6, alpha=0.05, prev_0=0.5):
    """
    ds = voxels x subjects x first_level_permutations
    
    % permutation-based prevalence inference using the minimum statistic, core
    % implementation of the method proposed by Allefeld, Goergen and Haynes (2016)
    %
    % [results, params] = prevalenceCore(a, P2 = 1e6, alpha = 0.05)
    %
    % a:            three-dimensional array of test statistic values
    %               (voxels x subjects x first-level permutations)
    %               a(:, :, 1) must contain actual values
    % P2:           number of second-level permutations to generate
    % alpha:        significance level
    % results:      per-voxel analysis results
    %   .puGN         uncorrected p-values for global null hypothesis         (Eq. 24)
    %   .pcGN         corrected p-values for global null hypothesis           (Eq. 26)
    %   .puMN         uncorrected p-values for majority null hypothesis       (Eq. 19)
    %   .pcMN         corrected p-values for majority null hypothesis         (Eq. 21)
    %   .gamma0u      uncorrected prevalence lower bounds                     (Eq. 20)
    %   .gamma0c      corrected prevalence lower bounds                       (Eq. 23)
    %   .aTypical     median values of test statistic where pcMN <= alpha     (Fig. 4b)
    % params:        analysis parameters and properties
    %   .V            number of voxels
    %   .N            number of subjects
    %   .P1           number of first-level permutations
    %   .P2           number of second-level permutations actually generated
    %   .alpha        significance level
    %   .puMNMin      smallest possible uncorrected p-value for majority H0
    %   .pcMNMin      smallest possible corrected p-value for majority H0
    %   .gamma0uMax   largest possible uncorrected prevalence lower bound
    %   .gamma0cMax   largest possible corrected prevalence lower bound       (Eq. 27)
    %
    % The 'majority null hypothesis' referenced here is a special case of the
    % prevalence null hypothesis (Eq. 17), where the critical value is gamma0 =
    % 0.5. It describes the situation where there is no effect in the majority
    % of subjects in the population. Rejecting it allows to infer that there is
    % an effect in the majority of subjects in the population. aTypical is only
    % defined where the (spatially extended) majority null hypothesis can be
    % rejected. Compare Fig. 4b and 'What does it mean for an effect to be
    % "typical" in the population?' in the Discussion of Allefeld, Goergen and
    % Haynes (2016).
    %
    % The function opens a figure window that shows results based on the
    % permutations generated so far and is regularly updated. If it is closed,
    % the computation is stopped (not aborted) and results are returned
    % based on the permutations so far.
    %
    % The window shows in the upper panels
    %  -- p-values for the majority null hypothesis for all voxels (blue dots),
    %  -- the currently smallest possible p-value (red line),
    %  -- and the significance threshold alpha (black line).
    % In the left side uncorrected values, puMN and puMNMin,
    % and on the right side corrected values, pcMN and pcMNMin.
    %
    % In the lower panels it shows
    %  -- prevalence lower bounds for all voxels (blue dots),
    %  -- the currently largest possible prevalence lower bound (red line),
    %  -- and the majority prevalence 0.5 (black line).
    % On the left side uncorrected values, gamma0u and gamma0uMax,
    % and on the right side corrected values, gamma0c and gamma0cMax.
    %
    % Upper and lower panels show complementary representations of the same
    % result, corresponding to algorithm Step 5a and 5b, respectively (see
    % paper).
    %
    % If many blue dots lie on the red line in either panel, this indicates
    % that a more differentiated result might be obtained from more
    % permutations. However, the amount of additional permutations needed for
    % that might be computationally unfeasible.
    %
    % If P2 is larger than the number of possible second-level permutations,
    % P1 ^ N, an error message recommends to set P2 = P1 ^ N. If this
    % recommendation is followed, Monte Carlo estimation is replaced by a
    % complete enumeration of all possible second-level permutations. In this
    % case, stopping the computation may bias the result.
    %
    % See also prevalence.
    %
    %
    % Copyright (C) 2016 Carsten Allefeld
    %
    % This program is free software: you can redistribute it and/or modify it
    % under the terms of the GNU General Public License as published by the
    % Free Software Foundation, either version 3 of the License, or (at your
    % option) any later version. This program is distributed in the hope that
    % it will be useful, but without any warranty; without even the implied
    % warranty of merchantability or fitness for a particular purpose. See the
    % GNU General Public License <http://www.gnu.org/licenses/> for more details.
    """


    n_voxels, n_subject, n_p1 = ds.shape
    
    if n_p2 > np.power(n_p1, n_subject):
        n_p2 = n_p1 ** n_subject
    
    
    full_permutation = (n_p2 == n_p1 ** n_subject)
    
    u_rank = np.zeros(n_voxels)
    c_rank = np.zeros(n_voxels)
    
    for j in range(n_p2):
        
            
        if j == 1:
            js = np.ones(n_subject)
        else:
            js = np.random.randint(n_p1, size=n_subject)
    

        m = np.min(ds[:, :, js])
        
    
    
        if j == 1:
            m1 = m
        
        
        u_rank += np.int_(m >= m1)
        c_rank += np.int_(np.max(m) >= m1)
    
    
    pu_GN = u_rank / n_p2
    pc_GN = c_rank / n_p2
    
    sig_GN = pu_GN <= alpha
    
    root_pn_M = root(pu_GN, n_subject)
    reciprocal_p2 = 1. / n_p2
    root_rec_p2 = root(reciprocal_p2, n_subject)
    
    pu_MN = np.power((1 - prev_0) * root_pn_M + prev_0, n_subject) # Eq. 19
    pc_MN = pc_GN + (1 - pc_GN) * pu_MN # Eq. 21
    
    sig_MN = (pc_MN <= alpha)
    
    
    pu_MN_min = ((1 - prev_0) * root(reciprocal_p2, n_subject) + prev_0) ** (n_subject)
    pc_MN_min = reciprocal_p2 + (1 - reciprocal_p2) * pu_MN_min
    
    gamma_0 = (root(alpha, n_subject) - root(pu_GN, n_subject)) / (1 - root(pu_GN,n_subject))
    gamma_0[alpha < pu_GN] = np.NaN
    
    alphac = (alpha - pc_GN) / (1 - pc_GN)
    
    gamma_0c = (root(alphac, n_subject) - root(pu_GN, n_subject)) / (1 - root(pu_GN, n_subject))
    gamma_0c[pu_GN < alphac] = np.NaN
    
    
    gamma_0u_max = (root(alpha, n_subject) - root(reciprocal_p2, n_subject)) / (1 - root_rec_p2)
    alpha_c_max = (alpha - reciprocal_p2) / (1 - reciprocal_p2) # Eq. 27
    gamma_0c_max = (root(alpha_c_max, n_subject) - root_rec_p2) / (1 - root_rec_p2)


    a_typical = np.zeros(n_voxels)
    a_typical[sig_MN] = np.median(ds[sig_MN, :, 1], 2)
    

    

def root(num, n):
    
    return np.power(num, 1./n)
    



def prob(num1, num2, g0):
    
    return (1 - g0) * num1 + g0 * num2




"""% where majority null hypothesis can be rejected, typical value of test statistic
aTypical = nan(V, 1);
aTypical(sigMN) = median(a(sigMN, :, 1), 2);

% collect return values
params = struct;
params.V = V;
params.N = N;
params.P1 = P1;
params.P2 = P2;
params.alpha = alpha;
params.puMNMin = puMNMin;
params.pcMNMin = pcMNMin;
params.gamma0uMax = gamma0uMax;
params.gamma0cMax = gamma0cMax;
results = struct;
results.puGN = puGN;
results.pcGN = pcGN;
results.puMN = puMN;
results.pcMN = pcMN;
results.gamma0u = gamma0u;
results.gamma0c = gamma0c;
results.aTypical = aTypical;"""

