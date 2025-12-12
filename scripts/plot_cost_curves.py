import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameters
    # User Preferences: 1.0 (Max Comfort), 0.5 (Balanced), 0.0 (Eco)
    user_prefs = [1.0, 0.5, 0.0]
    
    # X-Axis now represents "Bad Deviation (°F)" AND "Energy Usage (kWh)"
    # We plot from 0 to 6 to see the intersection points clearly
    x = np.linspace(0, 6, 300) 

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.title('Optimization Cost Trade-offs', fontsize=16)

    # 1. Plot Comfort Costs
    # User Pref -> Internal p_center: 0.1 + (user * 0.4)
    # Cost = p_center * (Deviation)^2
    for u_pref in user_prefs:
        p = 0.1 + (u_pref * 0.4)
        cost_comfort = p * (x ** 2)
        plt.plot(x, cost_comfort, linewidth=2, label=f'Comfort User={u_pref} (p={p:.2f})')

    # 2. Plot Energy Cost
    # Cost = Energy (kWh)
    # So y = x
    cost_energy = x
    plt.plot(x, cost_energy, 'r--', linewidth=2, label='Energy Cost (1 kWh = 1 Cost Unit)')

    # Styling
    plt.xlabel('Deviation (°F) / Energy (kWh)', fontsize=12)
    plt.ylabel('Cost (Equivalent)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add intersection points annotation (optional, but requested "so i can see where they intersect")
    # Intersection: p*x^2 = x  => p*x = 1 => x = 1/p
    # (ignoring x=0 trivial solution)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, u_pref in enumerate(user_prefs):
        p = 0.1 + (u_pref * 0.4)
        if p > 0:
            intersect_x = 1.0 / p
            intersect_y = intersect_x # since y=x at intersection
            
            # Check if within plot range
            if intersect_x <= 6:
                plt.plot(intersect_x, intersect_y, 'ko', markersize=6)
                plt.annotate(f'{intersect_x:.1f}', 
                             (intersect_x, intersect_y), 
                             xytext=(5, 5), textcoords='offset points')

    plt.xlim(0, 6)
    plt.ylim(0, 10) # Adjust Y limit to keep things readable

    # Save plot
    output_file = 'cost_curves.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
