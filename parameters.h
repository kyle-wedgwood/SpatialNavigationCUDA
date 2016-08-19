// connectivity parameters
#define N 1000
#define sigma  25.0f

// spatial parameters
#define L 250.0f

// alpha function
#define alpha 0.5f

// model parameters
#define tau 1.0f //time constant
#define tau_h 417.0f //time constant
#define tau_r 20.0f //refractoriness constant
#define I  0.01f //constant current
#define V_th 6.5f //threshold
#define V_r  0.0f //reset
#define W 1.0f // synaptic connection strength

// h-current parameters
#define gh  -3.0f*(-10.0f+65.0f) // 165.0f

// synaptic current parameters
#define gs -2.1f*(-85.0f+65.0f)

#define beta_left  0.94f
#define beta_right  -0.06f

#define V_left  -25.0f
#define V_right  5.0f
#define gamma_centre  (beta_right-beta_left)/(V_right-V_left) // -0.0333f
#define beta_centre (-gamma_centre*V_left+beta_left) // 0.1067f

#define tol 1e-6

// For plotting purposes
#define V_min -80.0f
