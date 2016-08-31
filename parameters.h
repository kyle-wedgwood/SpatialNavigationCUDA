// connectivity parameters
#define sigma  25.0f
#define w0 10.0f // synaptic connection strength
#define beta 0.5f
#define spatial_cutoff 1e-8

// REFFRACTORY TIMESCALE
#define tau_r 200.0f //refractoriness constant

// Voltage equation parameters
// CONDUCTANCES
#define gh_steve 1.0f
#define gs_steve 15.0f
#define gl_steve 0.25f

// TIMESCALES
#define C_steve 1.0f //time constant
#define tau_h_steve 400.0f //time constant

// INPUT
#define I_steve  0.01f //constant current

// RESET
#define V_th 14.5f //threshold
#define V_r  0.0f //reset

// REVERSAL POTENTIAL
#define V_h_steve 40.f

// RESCALING
#define tau (C_steve/gl_steve)
#define gs (-gs_steve/gl_steve)
#define gh (gh_steve*V_h_steve/gl_steve)
//#define I (-I_steve/gl_steve)

// Gating equation parameters

// TIMESCALE
#define tau_h 400.0f //time constant

// SWITCHES
#define V_half  -10.0f
#define V_k 10.0f
#define V_left (V_half-2*V_k)
#define V_right (V_half+2*V_k)

#define beta_left 1.0f
#define beta_right  0.0f
#define beta_centre (0.5f+V_half/(4*V_k))
#define gamma_centre (-1.0f/(4*V_k))

// ALPHA FUNCTION TIMESCALE
#define alpha 0.05f

// Newton solver
#define tol 1e-5

// APPLIED CURRENT PARAMETERS
#define I_app_first 0
#define I_app_last 250

// For plotting purposes
#define V_min -80.0f
