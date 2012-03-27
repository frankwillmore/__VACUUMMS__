/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                                                                          */
/*                          PMDGPU Version 1.0.0                            */
/*                                                                          */
/*                        (C) 2010 Frank T Willmore                         */
/*                                                                          */
/*                                                                          */
/*                                                                          */
/*  pmdgpu is a re-optimization of the utrwt69 algorithm published in:      */
/*                                                                          */
/*  "Gas diffusion in glasses via a probabilistic molecular dynamics."      */
/*  Willmore FT, Wang XY, Sanchez IC.                                       */
/*  JOURNAL OF CHEMICAL PHYSICS 126, 234502 JUNE 15 2007                    */
/*                                                                          */
/*  PMDGPU is developed by Frank T Willmore in collaboration with           */
/*  Isaac C Sanchez of the University of Texas at Austin department of      */
/*  chemical engineering.  Special thanks to Ying Jiang of the Sanchez      */
/*  research group for contributions to the project.                        */
/*                                                                          */
/*  PMDGPU Version 1.0.0 has been demonstrated to display statistically     */
/*  equivalent results for the diffusion of Helium in HAB6FDACl.  Results   */
/*  were compared for 10 configurations of HAB6FDACl, using n = 50          */
/*  insertions for each configuration for utrwt69 and for pmdgpu.  This     */
/*  was compared to 5 insertions into the same 10 configurations using the  */
/*  Accelrys Materials Studio simulation software package, and were         */
/*  verified to generate the same diffusivity value of 2.5e-04 sq cm/s.     */
/*                                                                          */
/*  Allocation suport on the resources Spur and Longhorn is provided by:    */
/*                                                                          */
/*    The Texas Advanced Computing Center                                   */
/*    The National Science Foundation/NSF-Teragrid                          */
/*                                                                          */
/*  correspondence to:  frankwillmore@gmail.com                             */
/*                                                                          */
/****************************************************************************/

Feb 15 2010:  Beginning process to extend functionality to linear molecules

