    def timestepping(self, tf):
        """Timesteps the physical problem with a set dt until user-specified time is reached"""
        dt = 0.1 * self.h / self.c0
        # print(dt)
        t = 0
        v_ij_max = 0
        a_max = 0
        rho_max_condition = 0
        assert (tf >= dt), "time to short to resolve problem, dt=%.4f"%(dt)

        count = 0
        while self.t_curr <= tf:
            print("Timestep iteration %g..."%(count + 1))

            # find all the derivatives for each particle
            for i, p_i in enumerate(self.particle_list):
                # create list of neighbours for particle i
                self.neighbour_iterate(p_i)
                
                # if not p_i.bound or 
                
                # calculate smoothing contribution from all neighbouring particles
                dW_i = self.dW(p_i, p_i.adj)

                # calculate acceleration and rate of change of density, find maximum relative velocity
                # amongst all particles and their neighbours and the maximum acceleration amongst particles
                p_i.a = self.g
                p_i.D = 0
                for j, p_j in enumerate(p_i.adj):
                    r_vec = p_i.x - p_j.x
                    r_mod = np.sqrt(np.sum(r_vec ** 2))
                    e_ij = r_vec / r_mod
                    v_ij = p_i.v - p_j.v
                    # print(p_i.v == p_j.v)

                    p_i.a -= p_j.m * (p_i.P / p_i.rho ** 2 + p_j.P / p_j.rho ** 2) * dW_i[j] * e_ij
                    p_i.a += self.mu * p_j.m * (1/p_i.rho**2 + 1/p_j.rho**2)*dW_i[j]*v_ij / r_mod

                    p_i.D += p_j.m * dW_i[j] * (v_ij[0]*e_ij[0] + v_ij[1]*e_ij[1])

                    v_ij_max = np.amax((np.linalg.norm(v_ij), v_ij_max))

                a_max = np.amax((np.linalg.norm(p_i.a), a_max))

                rho_condition = np.sqrt((p_i.rho/self.rho0)**(self.gamma-1))
                rho_max_condition = np.amax((rho_max_condition, rho_condition))

            # Updating the time step
            if count > 0:
                # cfl_dt = self.h / v_ij_max
                # f_dt = np.sqr/t(self.h / a_max)
                # a_dt = np.amin(self.h / (self.c0 * rho_max_condition))
                # dt = self.CFL * np.amin([cfl_dt, f_dt, a_dt])
                o = "hi"

            # updating each particles values
            for i, p_i in enumerate(self.particle_list):
                # update position -- needs to be updated before new velocity is computed
                p_i.x = p_i.x + dt * p_i.v

                # update velocity
                p_i.v = p_i.v + dt * p_i.a

                # update density, smooths if count is a multiple of smoothing
                p_i.rho = p_i.rho + dt * p_i.D
                if count % self.interval_smooth == 0:
                    p_j_list = p_i.adj[:]
                    p_j_list.append(p_i)
                    p_i.rho = self.rho_smoothing(p_i, p_j_list)

                # update pressure
                p_i.P = self.B * ((p_i.rho/self.rho0)**self.gamma - 1)

                # update particle indices
                p_i.calc_index()

            # re-allocate particles to grid
            self.allocate_to_grid()
            if count % self.interval_save:
                self.save_state()

            count += 1
            self.t_curr += dt
        self.file.close()
        return None